#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import cython
from cython.parallel import parallel, prange, threadid
import logging
from timeit import default_timer
from libc.stdlib cimport malloc, free

from scipy.stats import pearsonr,spearmanr

from collections import defaultdict, Counter
from libcpp.vector cimport vector
from libcpp.map cimport map as cpp_map
from libcpp.queue cimport queue
from libcpp.string cimport string
from libcpp.pair cimport pair

import numpy as np
cimport numpy as np

from libc.math cimport sqrt, exp, log, round
from scipy.linalg.cython_blas cimport sdot,saxpy,snrm2
import time
import threading



ctypedef np.ndarray ndarray
REAL = np.float32
ctypedef np.float32_t REAL_t

UINT = np.uint32
ctypedef np.uint32_t UINT_t


# for sampling (negative and frequent-word downsampling)
cdef UINT_t RAND_MAX = 2**32 - 1


cdef unsigned long long next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)


# this quick & dirty RNG apparently matches Java's (non-Secure)Random
# note this function side-effects next_random to set up the next number
cdef inline UINT_t random_int32(
        unsigned long long *next_random
        ) nogil:
    cdef UINT_t this_random = next_random[0] >> 16
    next_random[0] = (next_random[0] * <unsigned long long>25214903917ULL + 11) & 281474976710655ULL
    return this_random


cdef inline UINT_t rand(
        unsigned long long *next_random
        ) nogil:
    return random_int32(next_random)

cdef inline REAL_t random(
        unsigned long long *next_random
        ) nogil:
    return <REAL_t>random_int32(next_random)/<REAL_t>RAND_MAX

cdef inline UINT_t randint(
        const UINT_t high,
        unsigned long long *next_random
        ) nogil:
    return <UINT_t>(<REAL_t>random_int32(next_random)/<REAL_t>RAND_MAX  * (high - 1))

cdef UINT_t UONE = 1
cdef UINT_t UZERO = 0
cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0
cdef int ZERO = 0
cdef REAL_t ZEROF = <REAL_t>0.0
DEF EXP_TABLE_SIZE = 100000
DEF MAX_EXP = 10

# exp cache table
cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE
cdef REAL_t[EXP_TABLE_SIZE] TRUE_LOSS_TABLE
cdef REAL_t[EXP_TABLE_SIZE] FALSE_LOSS_TABLE


def init_variables():
    logging.info("initializing cython module")
    # build the sigmoid table
    cdef int i
    cdef REAL_t x
    for i in range(EXP_TABLE_SIZE):
        x = (i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP
        # exp(x)
        EXP_TABLE[i] = <REAL_t>exp(x)
        # calculate loss
        # loss function: max(x, 0) - x * z + log(1 + exp(-abs(x)))
        # from https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
        FALSE_LOSS_TABLE[i] = <REAL_t>(max(x,0) + log(1 + exp(-1 * abs(x))))
        TRUE_LOSS_TABLE[i] = <REAL_t>( -1 * x + FALSE_LOSS_TABLE[i] )

        # sigmoid : exp(x) / ( exp(x) + 1 )
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))
    logging.info("cython module initialized")

init = init_variables()


# def zeros_aligned(shape, dtype, order='C', align=128):
#     """Like `np.zeros()`, but the array will be aligned at `align` byte boundary."""
#     nbytes = np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize
#     buffer = np.zeros(nbytes + align, dtype=np.uint8)  # problematic on win64 ("maximum allowed dimension exceeded")
#     start_index = -buffer.ctypes.data % align
#     return buffer[start_index: start_index + nbytes].view(dtype).reshape(shape, order=order)



# to support random draws from negative-sampling cum_table
cdef inline UINT_t bisect_left(vector[UINT_t] *a, UINT_t x, UINT_t lo, UINT_t hi) nogil:
    cdef UINT_t mid
    while hi > lo:
        mid = (lo + hi) >> 1
        if a[0][mid] >= x:
            hi = mid
        else:
            lo = mid + 1
    return lo

cdef class paper2vec:
    # variables
    cdef public dict word2id
    cdef public dict paper2id
    cdef public ndarray id2word
    cdef public list id2paper


    cdef vector[UINT_t] w2v_frequence
    cdef vector[UINT_t] d2v_frequence
    cdef vector[UINT_t] d2v_word_frequence
    cdef vector[UINT_t] n2v_frequence

    cdef public ndarray w2v_id2rank
    cdef public ndarray w2v_rank2id

    cdef vector[REAL_t] w2v_subsample_table
    cdef vector[REAL_t] d2v_subsample_table
    cdef vector[REAL_t] n2v_subsample_table

    cdef vector[REAL_t] w2v_freq_i
    cdef vector[REAL_t] w2v_freq_o
    cdef vector[REAL_t] w2v_freq_n

    cdef vector[REAL_t] d2v_freq_i
    cdef vector[REAL_t] d2v_freq_o
    cdef vector[REAL_t] d2v_freq_n


    cdef vector[REAL_t] n2v_freq_i
    cdef vector[REAL_t] n2v_freq_o
    cdef vector[REAL_t] n2v_freq_n

    cdef vector[UINT_t] w2v_neg_table
    cdef vector[UINT_t] d2v_neg_table
    cdef vector[UINT_t] n2v_neg_table

    cdef vector[vector[UINT_t]] words
    cdef vector[vector[UINT_t]] graph
    cdef vector[vector[vector[UINT_t]]] n_order_neighbors
    cdef vector[UINT_t] word_draw_table
    cdef vector[UINT_t] node_draw_table
    cdef REAL_t domain
    cdef vector[UINT_t] current_log
    cdef vector[UINT_t] current_position
    cdef vector[int] reduced_window_table
    cdef unsigned long long next_random
    cdef ndarray this_batch_loss
    cdef unsigned long long cur_paper_id
    cdef unsigned long long cur_word_id
    cdef queue[UINT_t] cur_context
    cdef int shuffle


    # weights
    cdef public ndarray word_embeddings
    cdef public ndarray paper_embeddings
    cdef public ndarray word_output_weight
    cdef public ndarray paper_output_weight


    # parameters
    cdef int dimension
    cdef REAL_t alpha
    cdef REAL_t min_alpha

    cdef int w2v_window

    cdef int undirected

    cdef int negative

    cdef int w2v_min_count
    cdef int d2v_min_count
    cdef int n2v_min_count

    cdef int word_size
    cdef int paper_size

    cdef REAL_t w2v_subsampling
    cdef REAL_t d2v_subsampling
    cdef REAL_t n2v_subsampling

    cdef REAL_t noise_distribution

    cdef REAL_t w2v_ratio
    cdef REAL_t d2v_ratio
    cdef REAL_t n2v_ratio

    cdef int d2v_word_embeddings_output



    cdef int workers
    cdef unsigned long batch_size
    cdef unsigned long total_samples
    cdef unsigned long w2v_training_samples
    cdef unsigned long d2v_training_samples
    cdef unsigned long n2v_training_samples
    cdef unsigned long samples_per_iter
    cdef int iteration
    cdef int no_self_predict
    cdef int tficf
    cdef int tfidf
    cdef int LDE

    cdef REAL_t l2

    cdef REAL_t n2v_p
    cdef int n_order


    @property
    def dimension(self):
        return self.dimension

    @dimension.setter
    def dimension(self,value):
        self.dimension = value

    @property
    def w2v_frequence(self):
        return np.array(self.w2v_frequence)

    @property
    def d2v_word_frequence(self):
        return np.array(self.d2v_word_frequence)

    @property
    def d2v_frequence(self):
        return np.array(self.d2v_frequence)

    @property
    def d2v_id2word(self):
        return np.array(self.d2v_id2word)

    @property
    def d2v_word2id(self):
        return np.array(self.d2v_word2id)

    @property
    def n2v_frequence(self):
        return np.array(self.n2v_frequence)

    @property
    def w2v_window(self):
        return self.w2v_window

    @w2v_window.setter
    def w2v_window(self,value):
        self.w2v_window = value


    @property
    def graph(self):
        return self.graph

    @graph.setter
    def graph(self,value):
        self.graph = value

    @property
    def words(self):
        return self.words

    @words.setter
    def words(self,value):
        self.words = value



    @property
    def w2v_subsample_table(self):
        return np.array(self.w2v_subsample_table)

    @w2v_subsample_table.setter
    def w2v_subsample_table(self,value):
        self.w2v_subsample_table = value

    @property
    def d2v_subsample_table(self):
        return np.array(self.d2v_subsample_table)

    @d2v_subsample_table.setter
    def d2v_subsample_table(self,value):
        self.d2v_subsample_table = value

    @property
    def n2v_subsample_table(self):
        return np.array(self.n2v_subsample_table)

    @n2v_subsample_table.setter
    def n2v_subsample_table(self,value):
        self.n2v_subsample_table = value

    @property
    def alpha(self):
        return self.alpha

    @alpha.setter
    def alpha(self,value):
        self.alpha = value

    @property
    def min_alpha(self):
        return self.min_alpha

    @min_alpha.setter
    def min_alpha(self,value):
        self.min_alpha = value

    @property
    def undirected(self):
        return self.undirected

    @undirected.setter
    def undirected(self,value):
        self.undirected = value

    @property
    def w2v_min_count(self):
        return self.w2v_min_count

    @w2v_min_count.setter
    def w2v_min_count(self,value):
        self.w2v_min_count = value

    @property
    def d2v_min_count(self):
        return self.d2v_min_count

    @d2v_min_count.setter
    def d2v_min_count(self,value):
        self.d2v_min_count = value

    @property
    def d2v_min_count(self):
        return self.d2v_min_count

    @d2v_min_count.setter
    def d2v_min_count(self,value):
        self.d2v_min_count = value

    @property
    def n2v_min_count(self):
        return self.n2v_min_count

    @n2v_min_count.setter
    def n2v_min_count(self,value):
        self.n2v_min_count = value

    @property
    def w2v_subsampling(self):
        return self.w2v_subsampling

    @w2v_subsampling.setter
    def w2v_subsampling(self,value):
        self.w2v_subsampling = value

    @property
    def d2v_subsampling(self):
        return self.d2v_subsampling

    @d2v_subsampling.setter
    def d2v_subsampling(self,value):
        self.d2v_subsampling = value

    @property
    def n2v_subsampling(self):
        return self.n2v_subsampling

    @n2v_subsampling.setter
    def n2v_subsampling(self,value):
        self.n2v_subsampling = value

    @property
    def noise_distribution(self):
        return self.noise_distribution

    @noise_distribution.setter
    def noise_distribution(self,value):
        self.noise_distribution = value

    @property
    def w2v_ratio(self):
        return self.w2v_ratio

    @w2v_ratio.setter
    def w2v_ratio(self,value):
        self.w2v_ratio = value

    @property
    def d2v_ratio(self):
        return self.d2v_ratio

    @d2v_ratio.setter
    def d2v_ratio(self,value):
        self.d2v_ratio = value

    @property
    def n2v_ratio(self):
        return self.n2v_ratio

    @n2v_ratio.setter
    def n2v_ratio(self,value):
        self.n2v_ratio = value

    @property
    def total_samples(self):
        return self.total_samples

    @total_samples.setter
    def total_samples(self,value):
        self.total_samples = value

    @property
    def n2v_p(self):
        return self.n2v_p

    @n2v_p.setter
    def n2v_p(self,value):
        self.n2v_p = value

    @property
    def l2(self):
        return self.l2

    @l2.setter
    def l2(self,value):
        self.l2 = value

    @property
    def shuffle(self):
        return self.shuffle

    @shuffle.setter
    def shuffle(self,value):
        self.shuffle = value

    @property
    def no_self_predict(self):
        return self.no_self_predict

    @no_self_predict.setter
    def no_self_predict(self,value):
        self.no_self_predict = value

    @property
    def w2v_neg_table(self):
        return np.array(self.w2v_neg_table)

    @property
    def d2v_neg_table(self):
        return np.array(self.d2v_neg_table)

    @property
    def n2v_neg_table(self):
        return np.array(self.n2v_neg_table)

    @property
    def w2v_freq_i(self):
        return np.array(self.w2v_freq_i)

    @property
    def w2v_freq_o(self):
        return np.array(self.w2v_freq_o)

    @property
    def w2v_freq_n(self):
        return np.array(self.w2v_freq_n)

    @property
    def d2v_freq_i(self):
        return np.array(self.d2v_freq_i)

    @property
    def d2v_freq_o(self):
        return np.array(self.d2v_freq_o)

    @property
    def d2v_freq_n(self):
        return np.array(self.d2v_freq_n)

    @property
    def n2v_freq_i(self):
        return np.array(self.n2v_freq_i)

    @property
    def n2v_freq_o(self):
        return np.array(self.n2v_freq_o)

    @property
    def n2v_freq_n(self):
        return np.array(self.n2v_freq_n)

    @property
    def negative(self):
        return self.negative

    @negative.setter
    def negative(self,value):
        self.negative = value





    def word(self,word):
        if word in self.word2id:
            return self.word_embeddings[self.word2id[word]]
        else:
            logging.warning("word %s not exits, returning zero",word)
            return np.zeros(self.dimension,dtype=REAL)

    def paper(self,paper):
        if paper in self.paper2id:
            return self.paper_embeddings[self.paper2id[paper]]
        else:
            logging.warning("paper %s not exits, returning zero",paper)
            return np.zeros(self.dimension,dtype=REAL)

    def __cinit__(
        paper2vec self,
        data = None,
        int dimension = 100,
        int w2v_window = 5,
        REAL_t alpha = 0.025,
        REAL_t min_alpha = 0.0001,

        int negative = 5,

        int undirected = 1,

        int w2v_min_count = 0,
        int d2v_min_count = 0,
        int n2v_min_count = 0,

        REAL_t w2v_subsampling = 0,
        REAL_t d2v_subsampling = 0,
        REAL_t n2v_subsampling = 0,

        REAL_t noise_distribution = 0.75,

        REAL_t w2v_ratio = 1,
        REAL_t d2v_ratio = 1,
        REAL_t n2v_ratio = 1,

        unsigned long total_samples = 0,
        int iteration = 5,
        int workers = 1,
        unsigned long batch_size = int(1e5),
        REAL_t n2v_p = 0.75,
        int n_order = 0,
        REAL_t l2 = 0,
        shuffle = 1,
        no_self_predict = 1,
        tfidf = False,
        d2v_word_embeddings_output = False,
        int LDE = 0,
    ):
        cdef int i

        # init variables
        self.dimension = dimension
        self.w2v_window = w2v_window
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.negative = negative
        self.undirected = undirected
        self.d2v_word_embeddings_output = 1 if d2v_word_embeddings_output else 0

        self.w2v_min_count = w2v_min_count
        self.d2v_min_count = d2v_min_count
        self.n2v_min_count = n2v_min_count

        self.w2v_ratio = w2v_ratio
        self.d2v_ratio = d2v_ratio
        self.n2v_ratio = n2v_ratio

        self.w2v_subsampling = w2v_subsampling
        self.d2v_subsampling = d2v_subsampling
        self.n2v_subsampling = n2v_subsampling

        self.noise_distribution = noise_distribution

        self.total_samples = <int>total_samples
        self.iteration = <int>iteration
        self.no_self_predict = no_self_predict
        self.workers = workers
        self.batch_size = batch_size
        self.l2 = l2
        self.n2v_p = n2v_p
        self.n_order = n_order
        self.shuffle = 1 if shuffle else 0
        self.LDE = LDE
        if self.LDE:
            logging.info("use LDE")
        else:
            logging.info("use P2V")

        if tfidf:
            self.tfidf = 1
            self.tficf = 0
            logging.info("use TF-IDF")
        else:
            self.tfidf = 0
            self.tficf = 1
            logging.info("use TF-ICF")





        self.next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)

        if data:
            self.init_model(data)


    def init_model(self,data):
        logging.info("initilize the model")


        ratio = self.w2v_ratio + self.d2v_ratio + self.n2v_ratio
        if ratio == 0:
            raise("use {w,d,n}2v_ratio to indicate the part you want to train")
        self.w2v_ratio = self.w2v_ratio / ratio
        self.d2v_ratio = self.d2v_ratio / ratio
        self.n2v_ratio = self.n2v_ratio / ratio

        # map for word in the corpus
        word2id = {}
        id2word = []
        # map for paper in the dataset
        paper2id = {}
        id2paper = []
        # map for term in a document
        self.w2v_frequence = vector[UINT_t]()
        self.d2v_frequence = vector[UINT_t]()
        self.d2v_word_frequence = vector[UINT_t]()
        self.n2v_frequence = vector[UINT_t]()


        if hasattr(data,"w2v_counter") and hasattr(data,"d2v_counter") and hasattr(data,"n2v_counter"):
            w2v_counter = data.w2v_counter
            d2v_counter = data.d2v_counter
            n2v_counter = data.n2v_counter
        else:
            w2v_counter = defaultdict(int)
            d2v_counter = defaultdict(int)
            n2v_counter = defaultdict(int)
            logging.info("counting frequence")
            for paper in data:
                for word in paper.words:
                    # w2v counter
                    w2v_counter[word] += 1

                # d2v counter
                d2v_counter[paper.idx] += len(paper.words)

                # n2v counter
                n2v_counter[paper.idx] += 0
                for citation in paper.citations:
                    n2v_counter[paper.idx] += 1
                    if self.undirected:
                        n2v_counter[citation] += 1

            logging.info("done")

        logging.info("loading data")
        for paper in data:
            # min count
            if n2v_counter[paper.idx] < self.n2v_min_count:
                continue
            if d2v_counter[paper.idx] < self.d2v_min_count:
                continue

            # get pid
            try:
                pid = paper2id[paper.idx]
            except KeyError:
                pid = len(paper2id)
                paper2id[paper.idx] = pid
                id2paper.append(paper.idx)
                self.n2v_frequence.push_back(0)
                self.d2v_frequence.push_back(0)
                self.graph.push_back(vector[UINT_t]())
                self.words.push_back(vector[UINT_t]())

            self.d2v_frequence[pid] += len(paper.words)
            for word in paper.words:
                # word min count
                if w2v_counter[word] < self.w2v_min_count:
                    continue
                # get wid
                try:
                    wid = word2id[word]
                except KeyError:
                    wid = len(word2id)
                    word2id[word] = wid
                    id2word.append(word)
                    self.w2v_frequence.push_back(0)
                    # idf add one smooth
                    self.d2v_word_frequence.push_back(0)

                self.w2v_frequence[wid] += 1
                self.words[pid].push_back(wid)

            for word in set(paper.words):
                # word min count
                if w2v_counter[word] < self.w2v_min_count:
                    continue
                wid = word2id[word]
                self.d2v_word_frequence[wid] += 1


            for citation in paper.citations:
                # get pid
                try:
                    cid = paper2id[citation]
                except KeyError:
                    cid = len(paper2id)
                    paper2id[citation] = cid
                    id2paper.append(citation)
                    self.n2v_frequence.push_back(0)
                    self.d2v_frequence.push_back(0)
                    self.graph.push_back(vector[UINT_t]())
                    self.words.push_back(vector[UINT_t]())

                self.graph[pid].push_back(cid)
                self.n2v_frequence[pid] += 1
                if self.undirected:
                    self.graph[cid].push_back(pid)
                    self.n2v_frequence[cid] += 1

        del w2v_counter
        del d2v_counter
        del n2v_counter

        cdef int order,source,target
        # use exact star sampling
        if self.n_order:
            logging.info("preprocessing exact star")
            if self.paper_size > 1e5:
                logging.warining("The citation graph is too large, OOM may occur")

            for pid in range(self.graph.size()):
                self.n_order_neighbors.push_back([self.graph[pid]])
                seen = {}.fromkeys(self.graph[pid])
                for order in range(1,self.n_order):
                    neighbors = []
                    for source in self.n_order_neighbors[pid][order-1]:
                        for target in self.graph[source]:
                            if target not in seen:
                                neighbors.append(target)
                                seen[target] = None
                    self.n_order_neighbors[pid].push_back(neighbors)


        # normalize for zero frequence items -- avoid zero division error
        for i in range(<int>self.w2v_frequence.size()):
            self.w2v_frequence[i] = max(self.w2v_frequence[i] ,UONE)
        for i in range(<int>self.d2v_frequence.size()):
            self.d2v_frequence[i] = max(self.d2v_frequence[i] ,UONE)
        for i in range(<int>self.n2v_frequence.size()):
            self.n2v_frequence[i] = max(self.n2v_frequence[i] ,UONE)


#         self.graph = graph
#         self.graph = np.array([np.array(x,dtype=UINT) for x in graph])
#         self.words = words
#             self.words.push_back(<vector[UINT_t]>w)
#         self.words = np.array([np.array(x,dtype=UINT) for x in words])

        # map for word in the corpus
        self.word2id = word2id
        self.id2word = np.array(id2word)
        # map for paper in the dataset
        self.paper2id = paper2id
        self.id2paper = id2paper

        self.paper_size = self.n2v_frequence.size()
        self.word_size = self.w2v_frequence.size()

        self.w2v_rank2id = np.array(np.flipud(np.argsort(self.w2v_frequence)),dtype=UINT)
        self.w2v_id2rank = np.empty(self.word_size,dtype=UINT)
        for idx,rank in enumerate(self.w2v_rank2id):
            self.w2v_id2rank[rank] = idx
        logging.info("data contains %d papers, %d words" % (self.paper_size,self.word_size))

        self.init_variables()

    def init_variables(self):

        # for w2v down sampling
        if self.w2v_subsampling and self.w2v_ratio:
            logging.info("pre-processing down sampling for w2v")
            self.w2v_subsample_table = np.zeros(self.word_size).tolist()
            self.build_subsample_table(
                &self.w2v_frequence,
                &self.w2v_subsample_table,
                self.w2v_subsampling,
            )
            logging.info("done")

        # for d2v down sampling
        if self.d2v_subsampling and self.d2v_ratio:
            logging.info("pre-processing down sampling for d2v")
            self.d2v_subsample_table = np.zeros(self.paper_size).tolist()
            self.build_subsample_table(
                &self.w2v_frequence,
                &self.d2v_subsample_table,
                self.d2v_subsampling,
            )
            logging.info("done")

        # for n2v down sampling
        if self.n2v_subsampling and self.n2v_ratio:
            logging.info("pre-processing down sampling for n2v")
            self.n2v_subsample_table = np.zeros(self.paper_size).tolist()
            self.build_subsample_table(
                &self.n2v_frequence,
                &self.n2v_subsample_table,
                self.n2v_subsampling,
            )
            logging.info("done")
        # for w2v
        w2v_training_samples = 0
        w2v_total_samples = 0
        if self.w2v_ratio:
            self.w2v_freq_i = vector[REAL_t]()
            self.w2v_freq_o = vector[REAL_t]()
            self.w2v_freq_n = vector[REAL_t]()


            # build w2v_freq_i
            for i in range(<int>self.w2v_frequence.size()):
                # frequence after down sampling
                if self.w2v_subsampling:
                    self.w2v_freq_i.push_back(max(1,<REAL_t>self.w2v_frequence[i] * self.w2v_subsample_table[i]))
                else:
                    self.w2v_freq_i.push_back(max(1,<REAL_t>self.w2v_frequence[i]))
                # total samples before down sampling
                w2v_total_samples += self.w2v_frequence[i]
                # training samples after down sampling
                w2v_training_samples += self.w2v_freq_i[i]

            # build w2v_freq_n
            w2v_noise_count = np.array(self.w2v_frequence) ** self.noise_distribution
            w2v_noise_count = w2v_noise_count / w2v_noise_count.sum() * w2v_training_samples
            for i in range(<int>self.w2v_frequence.size()):
                self.w2v_freq_n.push_back(max(1,w2v_noise_count[i]))

            # build w2v_freq_o
            w2v_output_count = np.array(self.w2v_frequence)
            w2v_output_count = w2v_output_count / w2v_output_count.sum() * w2v_training_samples
            for i in range(<int>self.w2v_frequence.size()):
                self.w2v_freq_o.push_back(max(1,w2v_output_count[i]))

        # for d2v
        d2v_training_samples = 0
        d2v_total_samples = 0
        if self.d2v_ratio:
            self.d2v_freq_i = vector[REAL_t]()
            self.d2v_freq_o = vector[REAL_t]()
            self.d2v_freq_n = vector[REAL_t]()

            # build d2v_freq_i
            for i in range(<int>self.d2v_frequence.size()):
                # frequence after down sampling
                if self.d2v_subsampling:
                    self.d2v_freq_i.push_back(max(1,<REAL_t>self.d2v_frequence[i] * self.d2v_subsample_table[i]))
                else:
                    self.d2v_freq_i.push_back(max(1,<REAL_t>self.d2v_frequence[i]))
                # total samples before down sampling
                d2v_total_samples += self.d2v_frequence[i]
                # training samples after down sampling
                d2v_training_samples += self.d2v_freq_i[i]

            # build d2v_freq_n
            d2v_noise_count = np.array(self.w2v_frequence) ** self.noise_distribution
            d2v_noise_count = d2v_noise_count / d2v_noise_count.sum() * d2v_training_samples
            for i in range(<int>self.w2v_frequence.size()):
                self.d2v_freq_n.push_back(max(1,d2v_noise_count[i]))

            # build d2v_freq_o
            d2v_output_count = np.array(self.w2v_frequence)
            d2v_output_count = d2v_output_count / d2v_output_count.sum() * d2v_training_samples
            for i in range(<int>self.w2v_frequence.size()):
                self.d2v_freq_o.push_back(max(1,d2v_output_count[i]))


        # for n2v
        n2v_training_samples = 0
        n2v_total_samples = 0
        if self.n2v_ratio:
            self.n2v_freq_i = vector[REAL_t]()
            self.n2v_freq_o = vector[REAL_t]()
            self.n2v_freq_n = vector[REAL_t]()


            # build n2v_freq_i
            for i in range(<int>self.n2v_frequence.size()):
                # frequence after down sampling
                if self.n2v_subsampling:
                    self.n2v_freq_i.push_back(max(1,<REAL_t>self.n2v_frequence[i] * self.n2v_subsample_table[i]))
                else:
                    self.n2v_freq_i.push_back(max(1,<REAL_t>self.n2v_frequence[i]))
                # total samples before down sampling
                n2v_total_samples += self.n2v_frequence[i]
                # training samples after down sampling
                n2v_training_samples += self.n2v_freq_i[i]

            # build n2v_freq_n
            n2v_noise_count = np.array(self.n2v_frequence) ** self.noise_distribution
            n2v_noise_count = n2v_noise_count / n2v_noise_count.sum() * n2v_training_samples
            for i in range(<int>self.n2v_frequence.size()):
                self.n2v_freq_n.push_back(max(1,n2v_noise_count[i]))

            # build n2v_freq_o
            n2v_output_count = np.array(self.n2v_frequence)
            n2v_output_count = n2v_output_count / n2v_output_count.sum() * n2v_training_samples
            for i in range(<int>self.n2v_frequence.size()):
                self.n2v_freq_o.push_back(max(1,n2v_output_count[i]))


        self.w2v_training_samples = w2v_training_samples
        self.d2v_training_samples = d2v_training_samples
        self.n2v_training_samples = n2v_training_samples

        #self.samples_per_iter = self.w2v_ratio * ( w2v_training_samples * self.w2v_window ) + self.d2v_ratio * d2v_training_samples + self.n2v_ratio * n2v_training_samples
        #if self.total_samples == 0:
        #    self.total_samples = self.samples_per_iter * self.iteration


        # word neg sampling table
        if self.w2v_ratio or self.d2v_ratio:
            assert self.word_size > 0
            logging.info("pre-processing negative sampling for w2v")
            self.w2v_neg_table = np.zeros(self.word_size,dtype=UINT)
            self.build_bisect_table(
                &self.w2v_frequence,
                &self.w2v_neg_table,
                self.noise_distribution,
            )

        # paper word neg sampling table
        if self.d2v_ratio:
            assert self.word_size > 0 and self.paper_size > 0
            logging.info("pre-processing negative sampling for d2v")
            self.d2v_neg_table = np.zeros(self.word_size,dtype=UINT)
            self.build_bisect_table(
                &self.d2v_word_frequence,
                &self.d2v_neg_table,
                self.noise_distribution,
                #self.paper_size, # log
            )

        # paper neg sampling table
        if self.n2v_ratio:
            assert self.paper_size > 0
            logging.info("pre-processing negative sampling for n2v")
            self.n2v_neg_table = np.zeros(self.paper_size).tolist()
            self.build_bisect_table(
                &self.n2v_frequence,
                &self.n2v_neg_table,
                self.noise_distribution,
            )

        logging.info("init embeddings")
        # init input/output weights
        if self.w2v_ratio:
            self.word_embeddings = np.empty((self.word_size, self.dimension), dtype=REAL)
        if self.w2v_ratio or self.d2v_ratio:
            self.word_output_weight = np.zeros((self.word_size, self.dimension), dtype=REAL)
        if self.d2v_ratio or self.n2v_ratio:
            self.paper_embeddings = np.empty((self.paper_size, self.dimension), dtype=REAL)
        if self.n2v_ratio:
            self.paper_output_weight = np.zeros((self.paper_size, self.dimension), dtype=REAL)

        if self.w2v_ratio:
            self.init_weights(
                    <REAL_t *>(np.PyArray_DATA(self.word_embeddings)),
                    self.word_embeddings.shape[0],
                    self.word_embeddings.shape[1]
                    )
        if self.d2v_ratio or self.n2v_ratio:
            self.init_weights(
                    <REAL_t *>(np.PyArray_DATA(self.paper_embeddings)),
                    self.paper_embeddings.shape[0],
                    self.paper_embeddings.shape[1]
                    )

        logging.info("done")


    cdef void init_weights(
        paper2vec self,
        REAL_t *embeddings,
        unsigned long long I,
        unsigned long long J,
    ) nogil:
        cdef unsigned long long i
        cdef REAL_t half = 0.5

        for i in range(I*J):
            embeddings[i] = <REAL_t>( ( random(&self.next_random) - half )  / self.dimension )




    cdef void build_subsample_table(
        paper2vec self,
        vector[UINT_t] *frequence,
        vector[REAL_t] *table,
        REAL_t sub_sample,
    ) nogil:
        cdef unsigned long long sum_frequence = 0
        cdef int freq
        cdef unsigned long i
        cdef REAL_t tmp

        for i in range(frequence[0].size()):
            sum_frequence += frequence[0][i]

        sub_sample = sub_sample * sum_frequence
        for i in range(frequence[0].size()):
            freq = frequence[0][i]
            # probability of keeping the samples for current item
            tmp = sqrt(sub_sample/freq) if freq > 0 else 0
            tmp = min(tmp,1)
            table[0][i] = tmp





    cdef void build_bisect_table(
        paper2vec self,
        vector[UINT_t] *frequence,
        vector[UINT_t] *table,
        REAL_t power = 1,
        int log = 0,
        int min_count = 0,
    ):
        cdef int freq
        cdef unsigned long i
        cdef REAL_t tmp
        train_pow = 0
        cumulative = 0
        for i in range(frequence[0].size()):
            if power != 1:
                tmp = <REAL_t>frequence[0][i] ** power
            else:
                tmp = <REAL_t>frequence[0][i]
            if log > 0:
                # tmp = np.log(log) - tmp
                tmp = 1 / np.log(log/tmp)
            train_pow += tmp

        for i in range(frequence[0].size()):
            if power != 1:
                tmp = <REAL_t>frequence[0][i] ** power
            else:
                tmp = <REAL_t>frequence[0][i]
            if log > 0:
                #tmp = np.log(log) - tmp
                tmp = 1 / np.log(log/tmp)
            cumulative += tmp
            table[0][i] = <UINT_t>round( cumulative / train_pow * RAND_MAX)
        if i > 0:
            if table[0][i] != RAND_MAX:
                raise ValueError("bisect table not aligned, %s / %s" % (table[0][i], RAND_MAX))


    def train(
        paper2vec self,
        evaluate = None,
        data = None,
        report_delay = 1,
        workers = None,
        total_samples = None,
        batch_size = None,
        evaluation_interval = None,
    ):

        # init variables
        # build search table
        if self.w2v_ratio or self.d2v_ratio:
            if self.word_draw_table.empty():
                logging.info("building word draw table")
                self.word_draw_table = np.zeros(self.paper_size).tolist()
                self.build_bisect_table(
                    &self.d2v_frequence,
                    &self.word_draw_table,
                    1
                )

        if self.n2v_ratio:
            if self.node_draw_table.empty():
                logging.info("building node draw table")
                self.node_draw_table = np.zeros(self.paper_size).tolist()
                self.build_bisect_table(
                    &self.n2v_frequence,
                    &self.node_draw_table,
                    1
                )
            self.current_log = np.ones(self.paper_size,dtype=UINT).tolist()
            self.current_position = [x for x in range(self.paper_size)]




        self.reduced_window_table.clear()
        for i in range(1,self.w2v_window+1):
            for _ in range(i):
                self.reduced_window_table.push_back(i-self.w2v_window-1)
                self.reduced_window_table.push_back(self.w2v_window-i+1)

        if total_samples is not None and total_samples == 0:
            return
        else:
            self.total_samples = int(total_samples)

        if batch_size:
            self.batch_size = int(batch_size)

        if workers:
            self.workers = max(1,int(workers))

        if self.shuffle == 0:
            logging.info("shuffle disabled, force using one core to guarantee the training sequence")
            self.workers = 1
            self.cur_paper_id = 0
            self.cur_word_id = -1


        if evaluation_interval:
            evaluation_interval = int(evaluation_interval)
        else:
            evaluation_interval = self.batch_size



        loss = np.empty(self.batch_size,dtype=REAL)
        cdef REAL_t[:] loss_ptr = loss
        all_loss = []
        cdef int batch
        cdef int j,tid
        cdef REAL_t delta,alpha = self.alpha

        cdef unsigned long long trained = 0
        cdef unsigned long long last_report_samples = 0
        start = default_timer()
        next_report = report_delay
        last_report = start
        # start training

        # split job into batches
        batch = self.total_samples//self.batch_size
        delta = (self.alpha-self.min_alpha)/self.total_samples

        self.this_batch_loss = np.zeros(self.batch_size, dtype=REAL)

        elapsed = 0

        # distribute the batch to workers
        logging.info("starting trainig threads with %d samples" % self.total_samples)
        for i in range(batch):
            if trained % evaluation_interval == 0 and evaluate:
                msg = dict(
                        samples=trained,
                        iteration=float(trained)/float(self.samples_per_iter),
                        loss = self.this_batch_loss.mean(),
                        )
                evaluate(self,data=data,fast=True,msg=msg)

            # workers > 1, use multi threads
            if self.workers > 1:
                threads = [threading.Thread(target=self.training_thread,args=(
                    trained,
                    delta,
                    tid,
                    )) for tid in range(self.workers)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
            else:
                self.training_thread(
                    trained,
                    delta,
                    0,
                    )

            #logging.info("this batch finished")

            elapsed = default_timer() - start
            #all_loss.append(loss.mean())
            trained += self.batch_size
            last_report_samples += self.batch_size
            if elapsed >= next_report:
                logging.info("progress: %.2f%%, %dM samples trained, current loss %.4f, current speed %.2fM/s, overall speed %.2fM/s, ETA: %ds" %
                             (
                                 100.0 * trained / self.total_samples,
                                 trained / 1e6,
                                 self.this_batch_loss.mean(),
                                 last_report_samples / (default_timer() - last_report) / 1e6,
                                 trained / elapsed / 1e6,
                                 elapsed * ( <REAL_t>(self.total_samples - trained) / <REAL_t>trained )
                             )
                            )
                last_report = default_timer()
                last_report_samples = 0
                next_report = elapsed + report_delay
        logging.info("%d samples trained in %d seconds" % (trained,elapsed))
        return elapsed

    def training_thread(
        paper2vec self,
        unsigned long long trained,
        REAL_t delta,
        int tid,
        ):

        cdef:
            unsigned long long next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)
            unsigned long long *private_next_random = &next_random
            unsigned long long j
            unsigned long long end
            REAL_t [:] this_batch_loss = self.this_batch_loss
            REAL_t x,y,z,s

        # normalized sampling ratio
        s = self.w2v_ratio + self.d2v_ratio + self.n2v_ratio
        x = self.w2v_ratio / s
        y = self.d2v_ratio / s + x
        z = self.n2v_ratio / s + x + y


        with nogil:
            # offset j
            j = self.batch_size // self.workers * tid
            end = j + self.batch_size // self.workers if tid < self.workers - 1 else self.batch_size
            while j < end:
                this_batch_loss[j] = self._training_thread(
                    self.alpha-delta*(trained+j),
                    private_next_random,
                    0, # not infer
                    x,
                    y,
                    z,
                )
                j += 1

    cdef REAL_t _training_thread(
        paper2vec self,
        REAL_t alpha,
        unsigned long long *private_next_random,
        int infer,
        REAL_t x,
        REAL_t y,
        REAL_t z,
    ) nogil:

        cdef UINT_t pid,wid
        cdef UINT_t predict,target,cur
        cdef UINT_t i,j
        cdef UINT_t word,window
        cdef int neighbor
        cdef UINT_t start,end
        cdef REAL_t p
        cdef REAL_t loss

        cdef REAL_t *worker_t = <REAL_t *>malloc(sizeof(REAL_t)*self.dimension)
        for i in range(self.dimension):
            worker_t[i] = 0


        while True:
            # w2v
            p = random(private_next_random)
            if p <= x and self.w2v_ratio:
                # shuffle the training samples
                if self.shuffle:
                    pid = bisect_left(&self.word_draw_table, rand(private_next_random), 0, self.word_draw_table.size()) if infer == 0 else infer
                    if self.words[pid].empty():
                        continue
                    word = randint(self.words[pid].size(),private_next_random)
                    predict = self.words[pid][word]
                    # subsampling
                    if self.w2v_subsampling and self.w2v_subsample_table[predict] < random(private_next_random):
                        continue
                    neighbor = word + self.reduced_window_table[randint(self.reduced_window_table.size(),private_next_random)]
                    if neighbor < 0 or neighbor >= <int>self.words[pid].size():
                        continue
                    target = self.words[pid][neighbor]
                    if self.no_self_predict and predict == target:
                        continue
                # generate the training samples
                else:
                    # move to next word
                    if self.cur_context.empty():
                        self.cur_word_id += 1
                        if self.cur_word_id >= self.words[self.cur_paper_id].size():
                            # next paper
                            # reset words
                            self.cur_word_id = 0
                            # next paper that contains words
                            self.cur_paper_id += 1
                            # reset pid if needed
                            if self.cur_paper_id >= self.words.size():
                                self.cur_paper_id = 0
                        # get the dynamic window
                        window = randint(self.w2v_window,private_next_random) + 1
                        start = max(0,<long long>self.cur_word_id-window)
                        end = min(self.words[self.cur_paper_id].size(),self.cur_word_id+window+1)
                        for i in range(start,end):
                            self.cur_context.push(i)
                    # this should never be triggered
                    if self.cur_context.empty():
                        continue
                    pid = self.cur_paper_id
                    wid = self.cur_word_id
                    predict = self.words[pid][wid]
                    target = self.words[pid][self.cur_context.front()]
                    self.cur_context.pop()
                    #continue
                    if self.no_self_predict and predict == target:
                        continue

                # train
                if self.LDE == 0:
                    loss = fast_pair(
                        predict,
                        target,
                        0,
                        <REAL_t *>self.word_embeddings.data,
                        <REAL_t *>self.word_output_weight.data,
                        &self.w2v_neg_table,
                        &self.w2v_freq_i,
                        &self.w2v_freq_o,
                        &self.w2v_freq_n,
                        &self.d2v_freq_i,
                        &self.d2v_freq_o,
                        &self.d2v_freq_n,
                        &self.n2v_freq_i,
                        &self.n2v_freq_o,
                        &self.n2v_freq_n,
                        self.w2v_ratio,
                        self.d2v_ratio,
                        self.n2v_ratio,
                        self.dimension,
                        self.negative,
                        self.w2v_window,
                        -2*self.l2,
                        alpha,
                        worker_t,
                        0 if infer else 1,
                        private_next_random,
                    )
                else:
                    loss = LDE_fast_pair(
                        pid,
                        predict,
                        target,
                        0,
                        <REAL_t *>self.word_embeddings.data,
                        <REAL_t *>self.paper_embeddings.data,
                        &self.w2v_neg_table,
                        &self.w2v_freq_i,
                        &self.w2v_freq_o,
                        &self.w2v_freq_n,
                        &self.d2v_freq_i,
                        &self.d2v_freq_o,
                        &self.d2v_freq_n,
                        &self.n2v_freq_i,
                        &self.n2v_freq_o,
                        &self.n2v_freq_n,
                        self.w2v_ratio,
                        self.d2v_ratio,
                        self.n2v_ratio,
                        self.dimension,
                        self.negative,
                        self.w2v_window,
                        -2*self.l2,
                        alpha,
                        worker_t,
                        0 if infer else 1,
                        private_next_random,
                    )

                break

            # d2v
            elif p <= y and self.d2v_ratio:
                # LDE don't need this
                if self.LDE:
                    break
                pid = bisect_left(&self.word_draw_table, rand(private_next_random), 0, self.word_draw_table.size()) if infer == 0 else infer
                if self.words[pid].empty():
                    continue
                wid = randint(self.words[pid].size(),private_next_random)

                predict = pid
                target = self.words[pid][wid]
                # subsampling
                if self.d2v_subsampling and self.d2v_subsample_table[target] < random(private_next_random):
                    continue
                    #with gil:
                    #    raise NotImplementedError

                # train
                loss = fast_pair(
                    predict,
                    target,
                    1,
                    <REAL_t *>self.paper_embeddings.data,
                    <REAL_t *>self.word_embeddings.data if self.d2v_word_embeddings_output else <REAL_t *>self.word_output_weight.data,
                    &self.d2v_neg_table if self.tfidf else &self.w2v_neg_table,
                    &self.w2v_freq_i,
                    &self.w2v_freq_o,
                    &self.w2v_freq_n,
                    &self.d2v_freq_i,
                    &self.d2v_freq_o,
                    &self.d2v_freq_n,
                    &self.n2v_freq_i,
                    &self.n2v_freq_o,
                    &self.n2v_freq_n,
                    self.w2v_ratio,
                    self.d2v_ratio,
                    self.n2v_ratio,
                    self.dimension,
                    self.negative,
                    self.w2v_window,
                    -2*self.l2,
                    alpha,
                    worker_t,
                    0 if infer else 1,
                    private_next_random,
                )
                break
            # n2v
            elif self.n2v_ratio:
                pid = bisect_left(&self.node_draw_table, rand(private_next_random), 0, self.node_draw_table.size()) if infer == 0 else infer

                # subsampling
                if self.n2v_subsampling and self.n2v_subsample_table[pid] < random(private_next_random):
                    continue

                if self.n_order:
                    # loopback
                    if random(private_next_random) > self.n2v_p**self.current_log[pid] or self.current_log[pid] >= self.n_order:
                        self.current_log[pid] = 1

                    # walker reaches a deadend, loopback to the original node and continue
                    if self.n_order_neighbors[pid][self.current_log[pid]].size() == 0:
                        self.current_log[pid] = 1
                        continue

                    predict = pid
                    target = self.n_order_neighbors[pid][self.current_log[pid]][randint(self.n_order_neighbors[pid][self.current_log[pid]].size(),private_next_random)]

                    self.current_log[pid] += 1

                else:
                    # loopback
                    p = self.n2v_p**self.current_log[pid]
                    if random(private_next_random) > self.n2v_p**self.current_log[pid]:
                        self.current_log[pid] = 1
                        self.current_position[pid] = pid

                    # walker's position
                    cur = self.current_position[pid]
                    # walker reaches a deadend, loopback to the original node and continue
                    if self.graph[cur].empty():
                        self.current_log[pid] = 1
                        self.current_position[pid] = pid
                        continue

                    predict = pid
                    target = self.graph[cur][randint(self.graph[cur].size(),private_next_random)]

                    self.current_position[pid] = target
                    self.current_log[pid] += 1

                if self.no_self_predict and predict == target:
                    continue

                # train
                loss = fast_pair(
                    predict,
                    target,
                    2,
                    <REAL_t *>self.paper_embeddings.data,
                    <REAL_t *>self.paper_output_weight.data if self.LDE == 0 else <REAL_t *>self.paper_embeddings.data,
                    &self.n2v_neg_table,
                    &self.w2v_freq_i,
                    &self.w2v_freq_o,
                    &self.w2v_freq_n,
                    &self.d2v_freq_i,
                    &self.d2v_freq_o,
                    &self.d2v_freq_n,
                    &self.n2v_freq_i,
                    &self.n2v_freq_o,
                    &self.n2v_freq_n,
                    self.w2v_ratio,
                    self.d2v_ratio,
                    self.n2v_ratio,
                    self.dimension,
                    self.negative,
                    self.w2v_window,
                    -2*self.l2,
                    alpha,
                    worker_t,
                    0 if infer else 1,
                    private_next_random,
                )
                break
            #else:
            #    with gil:
            #        print("error!")
        free(worker_t)
        return loss


    def _load_save_model(
        paper2vec self,
        path,
        atype = 'load'
        ):
        import os,pickle
        if not os.path.isdir(path):
            os.makedirs(path)

        def to_npy(attr):
            np.save(
                    "%s/%s.npy" % (path,attr),
                    getattr(self, attr),
                    allow_pickle=True
                    )

        def from_npy(attr):
            setattr(
                        self,
                        attr,
                        np.load(
                            "%s/%s.npy" % (path,attr),
                            allow_pickle=True
                        )
                    )

        def to_pickle(attr):
            with open("%s/%s.pickle" % (path,attr),'wb') as f:
                pickle.dump(
                    getattr(self, attr),
                    f,
                    protocol=4
                    )

        def from_pickle(attr):
            print(attr)
            with open("%s/%s.pickle" % (path,attr),'rb') as f:
                setattr(
                        self,
                        attr,
                        pickle.load(f)
                        )


        for attr in [
                'word_embeddings',
                'word_output_weight',
                'paper_embeddings',
                'paper_output_weight',
                ]:
            if atype == 'load':
                from_npy(attr)
            else:
                to_npy(attr)

        for attr in [
                "word2id",
                "id2word",
                "paper2id",
                "id2paper",
                ]:
            if atype == 'load':
                from_pickle(attr)
            else:
                to_pickle(attr)

        if atype == 'load':
            with open("%s/%s.pickle" % (path,'parameters'),'rb') as f:
                for key,value in pickle.load(f).items():
                    setattr(
                            self,
                            key,
                            value
                            )
        else:
            params = {
                    key:getattr(self,key) for key in [
                        'dimension',
                        'negative',
                        'w2v_window',
                        'alpha',
                        'min_alpha',
                        'undirected',
                        'w2v_min_count',
                        'd2v_min_count',
                        'n2v_min_count',
                        'w2v_subsampling',
                        'd2v_subsampling',
                        'n2v_subsampling',
                        'noise_distribution',
                        'w2v_ratio',
                        'd2v_ratio',
                        'n2v_ratio',
                        'total_samples',
                        'n2v_p',
                        'l2',
                        'shuffle',
                        'no_self_predict',
                        ]
                    }
            with open("%s/%s.pickle" % (path,'parameters'),'wb') as f:
                pickle.dump(
                        params,
                        f,
                        protocol=4
                        )



    def load_model(self,path):
        self._load_save_model(
            path,
            atype = 'load'
            )

    def save_model(self,path):
        self._load_save_model(
            path,
            atype = 'save'
            )


    def add_papers(
        paper2vec self,
        dataset,
        emb = None
            ):

        idxs = {}
        papers = []
        for paper in dataset:
            if paper.idx in self.paper2id or paper.idx in idxs:
                continue
            else:
                papers.append(paper)
                idxs[paper.idx] = None

        for paper in papers:
            self.graph.push_back(vector[UINT_t]())
            self.words.push_back(vector[UINT_t]())

            self.d2v_freq_i.push_back(1)
            self.n2v_subsample_table.push_back(1)
            self.n2v_freq_i.push_back(1)


        self.current_log = np.concatenate([self.current_log,[1 for x in range(len(papers))]],axis=None)
        self.current_position = np.concatenate([self.current_position,[len(self.paper2id)+x for x in range(len(papers))]],axis=None)

        self.paper_embeddings = np.concatenate(
                (
                    self.paper_embeddings,
                    np.empty(shape=(len(papers),self.dimension),dtype=REAL)
                ),
                axis=0
            )


        for paper in papers:
            # add into map
            pid = len(self.paper2id)
            self.paper2id[paper.idx] = pid
            self.id2paper.append(paper.idx)

            # remove temp data if there is any
            self.graph[pid].clear()
            self.words[pid].clear()
            self.n2v_freq_i[pid] = 1
            self.d2v_freq_i[pid] = 1

            # inject embedding
            self.paper_embeddings[pid] = emb

            # build temp graph
            for citation in paper.citations:
                # skip citations that not in the graph
                if citation not in self.paper2id:
                    continue
                cid = self.paper2id[citation]
                self.graph[pid].push_back(cid)
                # here we ignore the link of cid => pid as old paper is unlikely to cite new paper
                if self.undirected:
                    self.graph[cid].push_back(pid)


            # build temp words
            for word in paper.words:
                # skip word that not in the vocabulary
                if word not in self.word2id:
                    continue
                wid = self.word2id[word]
                self.words[pid].push_back(wid)
                # here we ignore the link of cid => pid as old paper is unlikely to cite new paper

            self.n2v_freq_i[pid] = max(1,self.graph[pid].size())
            self.d2v_freq_i[pid] = max(1,self.words[pid].size())

            self._infer(pid)




    def infer(
        paper2vec self,
        paper,
        int total_samples = 0,
        int add = 0,
        REAL_t l2 = -1,
        ):

        cdef:
            unsigned long long pid
            REAL_t _l2 = self.l2

        if l2 >= 0:
            self.l2 = l2

        # use last idx as current pid
        if paper.idx in self.paper2id:
            pid = self.paper2id[paper.idx]
        else:
            pid = len(self.paper2id)

        # warm up data
        if self.paper_embeddings.shape[0] <= pid:
            logging.info("first time infer, warm up")
            self.graph.push_back(vector[UINT_t]())
            self.words.push_back(vector[UINT_t]())

            self.d2v_freq_i.push_back(1)

            self.n2v_subsample_table.push_back(1)
            self.n2v_freq_i.push_back(1)

            self.current_log = np.append(self.current_log,1)
            self.current_position = np.append(self.current_position,pid)

            self.paper_embeddings = np.concatenate(
                    (
                        self.paper_embeddings,
                        np.empty(shape=(1,self.dimension),dtype=REAL)
                    ),
                    axis=0
                )
            logging.info("done...")

        # remove temp data if there is any
        self.graph[pid].clear()
        self.words[pid].clear()
        self.n2v_freq_i[pid] = 1
        self.d2v_freq_i[pid] = 1
        # build temp graph
        for citation in paper.citations:
            # skip citations that not in the graph
            if citation not in self.paper2id:
                continue
            cid = self.paper2id[citation]
            self.graph[pid].push_back(cid)
            # here we ignore the link of cid => pid as old paper is unlikely to cite new paper
            if self.undirected:
                self.graph[cid].push_back(pid)


        # build temp words
        for word in paper.words:
            # skip word that not in the vocabulary
            if word not in self.word2id:
                continue
            wid = self.word2id[word]
            self.words[pid].push_back(wid)
            # here we ignore the link of cid => pid as old paper is unlikely to cite new paper

        self._infer(pid,total_samples)


        # clean up the graph
        if self.undirected:
            for citation in paper.citations:
                # skip citations that not in the graph
                if citation not in self.paper2id:
                    continue
                cid = self.paper2id[citation]
                if self.graph[cid][self.graph[cid].size()] == pid:
                    self.graph[cid].pop_back()


        self.l2 = _l2

        return self.paper_embeddings[pid]

    def _infer(
            paper2vec self,
            unsigned long long pid,
            int total_samples = 0,
            ):

        cdef:
            unsigned long long next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)
            unsigned long long *private_next_random = &next_random
            unsigned long long j
            REAL_t delta,x,y,z,s
            unsigned long long cid
            REAL_t _l2 = self.l2


        # initilize embedding
        for i in range(self.dimension):
            self.paper_embeddings[pid,i] = ( np.random.random() - 0.5 ) / self.dimension

        self.n2v_freq_i[pid] = max(1,self.graph[pid].size())
        self.d2v_freq_i[pid] = max(1,self.words[pid].size())


        s = self.w2v_ratio + self.d2v_ratio + self.n2v_ratio

        # w2v trainings
        if self.w2v_training_samples > 0:
            x = <REAL_t>self.total_samples * self.w2v_ratio / s * ( <REAL_t>self.words[pid].size() / <REAL_t>self.w2v_training_samples )
        else:
            x = 0
        # d2v trainings
        if self.d2v_training_samples > 0:
            y = <REAL_t>self.total_samples * self.d2v_ratio / s * ( <REAL_t>self.words[pid].size() / <REAL_t>self.d2v_training_samples )
        else:
            y = 0
        # n2v trainings
        if self.n2v_training_samples > 0:
            z = <REAL_t>self.total_samples * self.n2v_ratio / s * ( <REAL_t>self.graph[pid].size() / <REAL_t>self.n2v_training_samples )
        else:
            z = 0

        # calculate samples
        if total_samples <= 0:
            total_samples = int(y + z)

        # do not train word embeddings
        x = 0

        s = y + z
        assert s > 0

        y = y/s
        z = z/s



        delta = (self.alpha-self.min_alpha)/total_samples


        with nogil:
            # offset j
            j = 0
            while j < total_samples:
                self._training_thread(
                    self.alpha-delta*j,
                    private_next_random,
                    pid, # infer idx
                    x,
                    y,
                    z,
                )
                j += 1


    def word_analogy_3CosAdd(
            paper2vec self,
            UINT_t[:,:] questions,
            int limit = 30000,
            ):
        if limit == 0:
            limit = self.word_size

        cdef UINT_t index,i
        # get top n embeddings and normalize it
        cdef ndarray word_embeddings= self.word_embeddings[self.w2v_rank2id[:limit]]
        norms = np.sqrt((word_embeddings* word_embeddings).sum(axis=1))
        word_embeddings/= norms[:, np.newaxis]

        cdef REAL_t [:,:] word_embeddings_view = word_embeddings
        cdef REAL_t *embeddings_ptr = <REAL_t *>word_embeddings.data
        cdef ndarray guess = np.zeros(self.dimension, dtype=REAL)
        cdef REAL_t [:] guess_view = guess
        cdef REAL_t *guess_ptr= <REAL_t *>guess.data
        cdef UINT_t [:,:] ignore = np.zeros((questions.shape[0],3), dtype=UINT)
        cdef UINT_t [:] outputs = np.zeros(questions.shape[0], dtype=UINT)
        cdef UINT_t [:] id2rank = self.w2v_id2rank
        cdef UINT_t [:] rank2id = self.w2v_rank2id
        cdef UINT_t a,b,c,d,idx,tmp_idx


        cdef REAL_t[:] sims = np.empty(questions.shape[0],dtype=REAL)
        cdef REAL_t acc = 0, err = 0, tmp

        with nogil:
            for index in range(questions.shape[0]):
                a = questions[index,0]
                b = questions[index,1]
                c = questions[index,2]
                d = questions[index,3]
                # limit the vocab
                if id2rank[a] > limit or \
                   id2rank[b] > limit or \
                   id2rank[c] > limit or \
                   id2rank[d] > limit:
                    continue
                # ignore a,b,c in the query
                ignore[index,0] = id2rank[a]
                ignore[index,1] = id2rank[b]
                ignore[index,2] = id2rank[c]
                # expect output d
                outputs[index] = id2rank[d]
                # generate query
                for i in range(self.dimension):
                    guess_view[i] = \
                            word_embeddings_view[id2rank[b],i] - \
                            word_embeddings_view[id2rank[a],i] + \
                            word_embeddings_view[id2rank[c],i]
                sims[index] = -1
                # guess
                for idx in range(word_embeddings.shape[0]):
                    if idx == id2rank[a] or idx == id2rank[b] or idx == id2rank[c]:
                        continue
                    # cosine similairty
                    tmp = sdot(&self.dimension, &embeddings_ptr[idx*self.dimension], &ONE, guess_ptr, &ONE)
                    # record the largest one
                    if tmp > sims[index]:
                        sims[index] = tmp
                        tmp_idx = idx
                if tmp_idx == id2rank[d]:
                    acc += 1
                else:
                    err += 1

        return acc/(acc+err)



    def word_analogy_3CosMul(
            paper2vec self,
            UINT_t[:,:] questions,
            int limit = 30000,
            REAL_t epsilon = 0.001,
            ):
        if limit == 0:
            limit = self.word_size

        cdef UINT_t index,i
        # get top n embeddings and normalize it
        cdef ndarray word_embeddings= self.word_embeddings[self.w2v_rank2id[:limit]]
        norms = np.sqrt((word_embeddings* word_embeddings).sum(axis=1))
        word_embeddings/= norms[:, np.newaxis]

        cdef REAL_t [:,:] word_embeddings_view = word_embeddings
        cdef REAL_t *embeddings_ptr = <REAL_t *>word_embeddings.data
        cdef UINT_t [:] id2rank = self.w2v_id2rank
        cdef UINT_t [:] rank2id = self.w2v_rank2id
        cdef UINT_t a,b,c,d
        cdef UINT_t tmp_idx,idx
        cdef REAL_t tmp

        cdef REAL_t[:] sims = np.empty(questions.shape[0],dtype=REAL)
        sims[:] = -1

        cdef REAL_t acc = 0, err = 0
        with nogil:
            for index in range(questions.shape[0]):
                a = questions[index,0]
                b = questions[index,1]
                c = questions[index,2]
                d = questions[index,3]
                # limit the vocab
                if id2rank[a] > limit or \
                   id2rank[b] > limit or \
                   id2rank[c] > limit or \
                   id2rank[d] > limit:
                    continue
                sims[index] = -1
                # query
                for idx in range(word_embeddings_view.shape[0]):
                    if idx == id2rank[a] or idx == id2rank[b] or idx == id2rank[c]:
                        continue
                    tmp = (
                            ( sdot(&self.dimension, &embeddings_ptr[idx*self.dimension], &ONE, &embeddings_ptr[id2rank[b]*self.dimension], &ONE) + 1.0 ) / 2.0 * ( sdot(&self.dimension, &embeddings_ptr[idx*self.dimension], &ONE, &embeddings_ptr[id2rank[c]*self.dimension], &ONE) + 1.0 ) / 2.0
                          ) / (
                            ( sdot(&self.dimension, &embeddings_ptr[idx*self.dimension], &ONE, &embeddings_ptr[id2rank[a]*self.dimension], &ONE) + 1.0 ) / 2.0 + epsilon
                          )
                    if tmp > sims[index]:
                        sims[index] = tmp
                        tmp_idx = idx
                # expect output d
                if tmp_idx == id2rank[d]:
                    acc += 1
                else:
                    err += 1

        return acc/(acc+err)



    def save_word_embeddings(
            paper2vec self,
            filename,
            binary = False
            ):
        logging.info("save word embeddings to %s" % filename)
        if binary:
            with open(filename,'wb') as f:
                f.write("%d %d\n".encode('utf8') % (self.word_size,self.dimension))
                for idx in range(self.word_size):
                    idx = self.w2v_rank2id[idx]
                    f.write(self.id2word[idx].encode("utf8") + b" " + self.word_embeddings[idx].tostring())
        else:
            with open(filename,'w') as f:
                f.write("%d %d\n" % (self.word_size,self.dimension))
                for idx in range(self.word_size):
                    idx = self.w2v_rank2id[idx]
                    f.write(self.id2word[idx])
                    f.write(" ")
                    f.write(" ".join(["%.8f" % x for x in self.word_embeddings[idx]]))
                    f.write("\n")
        logging.info("done")



    def save_paper_embeddings(
            paper2vec self,
            filename,
            binary = False
            ):
        logging.info("save paper embeddings to %s" % filename)
        if binary:
            with open(filename,'wb') as f:
                f.write("%d %d\n".encode('utf8') % (self.paper_size,self.dimension))
                for idx in range(self.paper_size):
                    f.write(self.id2paper[idx].encode("utf8") + b" " + self.paper_embeddings[idx].tostring())
        else:
            with open(filename,'w') as f:
                f.write("%d %d\n" % (self.paper_size,self.dimension))
                for idx in range(self.paper_size):
                    f.write(self.id2paper[idx])
                    f.write(" ")
                    f.write(" ".join(["%.8f" % x for x in self.paper_embeddings[idx]]))
                    f.write("\n")
        logging.info("done")

    def evaluate_sim(
            paper2vec self,
            sims,
            method = "word",
            normalize = True,
            **args
            ):

        def cosine_similarity(a,b,normalization = True):
            if normalization:
                a = a / np.sqrt((a ** 2).sum(-1))
                b = b / np.sqrt((b ** 2).sum(-1))
            return np.dot(a,b.T)

        Y_true = []
        Y_pred = []
        cdef REAL_t sim,wxwy,cxcy,wxcy,cxwy,wxcx,wyxy
        if method == "word":
            for source,target,sim in sims:
                Y_true.append(sim)
                Y_pred.append(cosine_similarity(
                    self.word_embeddings[self.word2id[source]],
                    self.word_embeddings[self.word2id[target]],
                    normalization = normalize
                ))
        elif method == "context":
            for source,target,sim in sims:
                Y_true.append(sim)
                Y_pred.append(cosine_similarity(
                    self.word_output_weight[self.word2id[source]],
                    self.word_output_weight[self.word2id[target]],
                    normalization = normalize
                ))
        elif method == "word+context":
            for source,target,sim in sims:
                Y_true.append(sim)
                wxwy = cosine_similarity(
                    self.word_embeddings[self.word2id[source]],
                    self.word_embeddings[self.word2id[target]],
                    normalization = True
                )
                cxcy = cosine_similarity(
                    self.word_output_weight[self.word2id[source]],
                    self.word_output_weight[self.word2id[target]],
                    normalization = True
                )
                wxcy = cosine_similarity(
                    self.word_embeddings[self.word2id[source]],
                    self.word_output_weight[self.word2id[target]],
                    normalization = True
                )
                cxwy = cosine_similarity(
                    self.word_output_weight[self.word2id[source]],
                    self.word_embeddings[self.word2id[target]],
                    normalization = True
                )
                wxcx = cosine_similarity(
                    self.word_embeddings[self.word2id[source]],
                    self.word_output_weight[self.word2id[source]],
                    normalization = True
                )
                wycy = cosine_similarity(
                    self.word_embeddings[self.word2id[target]],
                    self.word_output_weight[self.word2id[target]],
                    normalization = True
                )
                Y_pred.append(
                    ( wxwy + cxcy + wxcy + cxwy ) / (2 * np.sqrt( wxcx + 1 ) * np.sqrt( wycy + 1 ) )
                )
        else:
            raise
#        elif method == "cosine_output":
#            for source,target,sim in sims:
#                Y_true.append(sim)
#                Y_pred.append(cosine_similarity(
#                    self.word_embeddings[self.word2id[source]],
#                    self.word_embeddings[self.word2id[target]],
#                ))
        #r1 =  pearsonr(Y_pred,Y_true)[0]
        try:
            r1 = spearmanr(Y_pred,Y_true)[0]
        except ValueError:
            r1 = 0

        return r1





    def estimate_O(
            paper2vec self,
            int samples = int(1e5),
            **argv
            ):
        cdef ndarray p = np.zeros(samples,dtype=REAL)
        cdef REAL_t [:] p_list = p
        cdef int pid,word,predict,neighbor,target
        cdef int counter = 0
        cdef int i, j
        cdef REAL_t *word_embeddings = <REAL_t *>self.word_embeddings.data
        cdef REAL_t *word_output_weight = <REAL_t *>self.word_output_weight.data
        cdef unsigned long long input_index,output_index

        cdef unsigned long long next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)
        cdef unsigned long long *private_next_random = &next_random
        cdef REAL_t f

        with nogil:
            for i in range(samples):
                while True:
                    pid = bisect_left(&self.word_draw_table, rand(private_next_random), 0, self.word_draw_table.size())
                    if self.words[pid].empty():
                        continue
                    word = randint(self.words[pid].size(),private_next_random)
                    predict = self.words[pid][word]
                    # subsampling
                    if self.w2v_subsampling and self.w2v_subsample_table[predict] < random(private_next_random):
                        continue
                    neighbor = word + self.reduced_window_table[randint(self.reduced_window_table.size(),private_next_random)]
                    if neighbor < 0 or neighbor >= <int>self.words[pid].size():
                        continue
                    target = self.words[pid][neighbor]
                    #if predict == target:
                    #    continue


                    input_index = predict * self.dimension
                    output_index = target * self.dimension

                    f = sdot(&self.dimension, &word_embeddings[input_index], &ONE, &word_output_weight[output_index], &ONE)
                    p_list[i] = log(1/(1+exp(-f)))
                    for j in range(self.negative):
                        while True:
                            target = bisect_left(&self.w2v_neg_table, rand(private_next_random), 0, self.w2v_neg_table.size())
                            output_index = target * self.dimension

                            f = -sdot(&self.dimension, &word_embeddings[input_index], &ONE, &word_output_weight[output_index], &ONE)
                            # continue if dot product is not in range
                            if f < -MAX_EXP or f > MAX_EXP:
                                continue

                            # fast sigmod
                            # this line can change to ReLU if needed
                            f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

                            p_list[i] += log(1/(1+exp(-f)))
                            break

                    break
        return p.mean()/(1+self.negative)

def batch_query(
    ndarray embeddings,
    ndarray inputs,
    UINT_t[:,:] ignore,
    UINT_t[:] idxs,
    REAL_t[:] sims,
    int workers,
    int tid,
):
    cdef int i = 0,j = 0,k = 0,end = 0
    cdef int embeddings_index = 0
    cdef int inputs_index = 0
    cdef int dimension = embeddings.shape[1]
    cdef int vocab = embeddings.shape[0]
    cdef int size = idxs.shape[0]
    cdef REAL_t *embeddings_ptr = <REAL_t *>embeddings.data
    cdef REAL_t *inputs_ptr= <REAL_t *>inputs.data

    cdef REAL_t tmp
    # offline the inputs base on tid
    i = inputs.shape[0] // workers * tid
    end = i + inputs.shape[0] // workers if tid < workers - 1 else inputs.shape[0]
    with nogil:
        while i < end:
            tmp = -1
            # input query
            inputs_index = i*dimension
            for j in range(embeddings.shape[0]):
                # ignore a,b,c
                if j == ignore[i][0]:
                    continue
                elif j == ignore[i][1]:
                    continue
                elif j == ignore[i][2]:
                    continue
                else:
                    embeddings_index = j*dimension
                    # cosine similairty
                    tmp = sdot(&dimension, &embeddings_ptr[embeddings_index], &ONE, &inputs_ptr[inputs_index], &ONE)
                    # record the largest one
                    if tmp > sims[i]:
                        sims[i] = tmp
                        idxs[i] = j
            i += 1

cdef inline REAL_t fast_pair(
    UINT_t input_index,
    UINT_t output_index,
    int flag,
    REAL_t *input_weight,
    REAL_t *output_weight,
    vector[UINT_t] *neg_table,
    vector[REAL_t] *w2v_freq_i,
    vector[REAL_t] *w2v_freq_o,
    vector[REAL_t] *w2v_freq_n,
    vector[REAL_t] *d2v_freq_i,
    vector[REAL_t] *d2v_freq_o,
    vector[REAL_t] *d2v_freq_n,
    vector[REAL_t] *n2v_freq_i,
    vector[REAL_t] *n2v_freq_o,
    vector[REAL_t] *n2v_freq_n,

    REAL_t x,
    REAL_t y,
    REAL_t z,

    int size,
    int negative,
    int w2v_window,

    REAL_t l2,

    REAL_t alpha,
    REAL_t *work,
    int learn_output,
    unsigned long long *next_random
    ) nogil:

    # transfer variables into c types under gil
    cdef REAL_t f, g, label
    cdef UINT_t target_index
    cdef int d, i
    cdef REAL_t input_lambda, output_lambda, tmp, loss = 0
    # private work memory
    for i in range(size):
        work[i] = 0

    # index
    cdef long long input_vector_index = input_index * size, output_vector_index

    for d in range(negative+1):
        if d == 0:
            target_index = output_index
            label = ONEF
        else:
            target_index = bisect_left(&neg_table[0], rand(next_random), 0, neg_table[0].size())

            # false negative sample
            if target_index == input_index:
                continue
            label = ZEROF

        # output vector
        output_vector_index = target_index * size

        # logist
        f = sdot(&size, &input_weight[input_vector_index], &ONE, &output_weight[output_vector_index], &ONE)
        # continue if dot product is not in range
        if f < -MAX_EXP or f > MAX_EXP:
            continue

        # calculate loss
        if d == 0:
            loss += TRUE_LOSS_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        else:
            loss += FALSE_LOSS_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

        # fast sigmod
        # this line can change to ReLU if needed
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

        # gradient
        g = (label - f) * alpha

        # cross entropy
        # learn from output weight and save it in work
        saxpy(&size, &g, &output_weight[output_vector_index], &ONE, work, &ONE)

        if learn_output:
            # update output weight
            if l2 < 0:
                if flag == 0:
                    if y:
                        # dirty fix for l2 overflow
                        tmp = y / x * ( d2v_freq_o[0][target_index] + d2v_freq_n[0][target_index] * <REAL_t>negative ) + <REAL_t>w2v_window * ( w2v_freq_o[0][target_index] + w2v_freq_n[0][target_index] * <REAL_t>negative )
                        tmp = tmp if tmp > 1 else 1
                        output_lambda = alpha * l2 / tmp
                    else:
                        # dirty fix for l2 overflow
                        tmp = <REAL_t>w2v_window * ( w2v_freq_o[0][target_index] + w2v_freq_n[0][target_index] * <REAL_t>negative )
                        tmp = tmp if tmp > 1 else 1
                        output_lambda = alpha * l2 / tmp
                elif flag == 1:
                    if x:
                        # dirty fix for l2 overflow
                        tmp = ( d2v_freq_o[0][target_index] + d2v_freq_n[0][target_index] * <REAL_t>negative ) + x / y * <REAL_t>w2v_window * ( w2v_freq_o[0][target_index] + w2v_freq_n[0][target_index] * <REAL_t>negative )
                        tmp = tmp if tmp > 1 else 1
                        output_lambda = alpha * l2 / tmp
                    else:
                        # dirty fix for l2 overflow
                        tmp = d2v_freq_o[0][target_index] + d2v_freq_n[0][target_index] * <REAL_t>negative
                        tmp = tmp if tmp > 1 else 1
                        output_lambda = alpha * l2 / tmp
                elif flag == 2:
                    output_lambda = alpha * l2 / ( <REAL_t>n2v_freq_o[0][target_index] + <REAL_t>n2v_freq_n[0][target_index] * <REAL_t>negative )
                saxpy(&size, &output_lambda, &output_weight[output_vector_index], &ONE, &output_weight[output_vector_index], &ONE)
            saxpy(&size, &g, &input_weight[input_vector_index], &ONE, &output_weight[output_vector_index], &ONE)


    # push update to input_weights
    if l2 < 0:
        if flag == 0:
            input_lambda = alpha * x * l2 / ( x * w2v_freq_i[0][input_index] * w2v_window )
        elif flag == 1:
            if z:
                input_lambda = alpha * y * l2 / ( y * d2v_freq_i[0][input_index] + z * n2v_freq_i[0][input_index] )
            else:
                input_lambda = alpha * l2 / d2v_freq_i[0][input_index]
        elif flag == 2:
            if y:
                input_lambda = alpha * z * l2 / ( y * d2v_freq_i[0][input_index] + z * n2v_freq_i[0][input_index] )
            else:
                input_lambda = alpha * l2 /  n2v_freq_i[0][input_index]

        saxpy(&size, &input_lambda, &input_weight[input_vector_index], &ONE, work, &ONE)
        loss -= input_lambda * sdot(&size, &input_weight[input_vector_index], &ONE, &input_weight[input_vector_index], &ONE)

    saxpy(&size, &ONEF, work, &ONE, &input_weight[input_vector_index], &ONE)
    # normalize loss
    return loss/(1+negative)


cdef inline REAL_t LDE_fast_pair(
    UINT_t paper_index,
    UINT_t input_index,
    UINT_t output_index,
    int flag,
    REAL_t *word_weight,
    REAL_t *paper_weight,
    vector[UINT_t] *neg_table,
    vector[REAL_t] *w2v_freq_i,
    vector[REAL_t] *w2v_freq_o,
    vector[REAL_t] *w2v_freq_n,
    vector[REAL_t] *d2v_freq_i,
    vector[REAL_t] *d2v_freq_o,
    vector[REAL_t] *d2v_freq_n,
    vector[REAL_t] *n2v_freq_i,
    vector[REAL_t] *n2v_freq_o,
    vector[REAL_t] *n2v_freq_n,

    REAL_t x,
    REAL_t y,
    REAL_t z,

    int size,
    int negative,
    int w2v_window,

    REAL_t l2,

    REAL_t alpha,
    REAL_t *work,
    int learn_output,
    unsigned long long *next_random
    ) nogil:

    # transfer variables into c types under gil
    cdef REAL_t f1, f2, f, g, label
    cdef UINT_t target_index
    cdef UINT_t d, i
    cdef REAL_t input_lambda, output_lambda, tmp, loss = 0
    # private work memory
    for i in range(size):
        work[i] = 0

    # index
    cdef long long input_vector_index = input_index * size, output_vector_index
    cdef long long paper_vector_index = paper_index * size

    for d in range(negative+1):
        if d == 0:
            target_index = output_index
            label = ONEF
        else:
            target_index = bisect_left(&neg_table[0], rand(next_random), 0, neg_table[0].size())

            # false negative sample
            if target_index == input_index:
                continue
            label = ZEROF

        # output vector
        output_vector_index = target_index * size

        # logist
        f1 = sdot(&size, &word_weight[input_vector_index], &ONE, &word_weight[output_vector_index], &ONE)
        f2 = sdot(&size, &paper_weight[paper_vector_index], &ONE, &word_weight[output_vector_index], &ONE)
        f = f1+f2
        # continue if dot product is not in range
        if f < -MAX_EXP or f > MAX_EXP:
            continue

        # calculate loss
        if d == 0:
            loss += TRUE_LOSS_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        else:
            loss += FALSE_LOSS_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

        # fast sigmod
        # this line can change to ReLU if needed
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

        # gradient
        g = (label - f) * alpha

        # cross entropy
        # learn from output weight and save it in work
        saxpy(&size, &g, &word_weight[output_vector_index], &ONE, work, &ONE)

        if learn_output:
            # update output weight
            if l2 < 0:
                # dirty fix for l2 overflow
                tmp = <REAL_t>w2v_window * ( w2v_freq_o[0][target_index] + w2v_freq_n[0][target_index] * <REAL_t>negative )
                tmp = tmp if tmp > 1 else 1
                output_lambda = alpha * l2 / tmp
                # learn l2
                saxpy(&size, &output_lambda, &word_weight[output_vector_index], &ONE, &word_weight[output_vector_index], &ONE)
            # learn from word
            saxpy(&size, &g, &word_weight[input_vector_index], &ONE, &word_weight[output_vector_index], &ONE)
            # learn from paper
            saxpy(&size, &g, &paper_weight[paper_vector_index], &ONE, &word_weight[output_vector_index], &ONE)


    # push update to input_weights
    if l2 < 0:
        # input word lambda
        input_lambda = alpha * x * l2 / ( x * w2v_freq_i[0][input_index] * w2v_window )
        saxpy(&size, &input_lambda, &word_weight[input_vector_index], &ONE, work, &ONE)

        # input paper lambda
        input_lambda = alpha * x * l2 / ( x * d2v_freq_i[0][paper_index] * w2v_window + z * n2v_freq_i[0][paper_index])
        saxpy(&size, &input_lambda, &paper_weight[paper_vector_index], &ONE, work, &ONE)

        loss -= input_lambda * sdot(&size, &word_weight[input_vector_index], &ONE, &word_weight[input_vector_index], &ONE)

    saxpy(&size, &ONEF, work, &ONE, &word_weight[input_vector_index], &ONE)
    saxpy(&size, &ONEF, work, &ONE, &paper_weight[paper_vector_index], &ONE)
    # normalize loss
    return loss/(1+negative)



import logging
from collections import defaultdict, namedtuple
import sys
import pickle
import os


def file_len(fname):
    import subprocess
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

class DataIterator():
    def __init__(self,data_path='./',debug=False):
        self.path = data_path
        self.labels = defaultdict(list)
        self.debug = debug

        length = file_len(self.path+"/idxs.txt")
        fidx = open(self.path+"/idxs.txt")

        if os.path.exists(self.path+"/w2v_counter.pickle") and os.path.exists(self.path+"/d2v_counter.pickle") and os.path.exists(self.path+"/n2v_counter.pickle"):
            import pickle
            with open(self.path+"/w2v_counter.pickle",'rb') as f:
                self.w2v_counter = pickle.load(f)
            with open(self.path+"/d2v_counter.pickle",'rb') as f:
                self.d2v_counter = pickle.load(f)
            with open(self.path+"/n2v_counter.pickle",'rb') as f:
                self.n2v_counter = pickle.load(f)

        #if os.path.exists(self.path+"/labels.pickle"):
        #    with open(self.path+"/labels.pickle",'rb') as f:
        #        self.labels = pickle.load(f)
        #elif os.path.exists(self.path+"/labels.txt"):
        #    flabels = open(self.path+"/labels.txt")
        #    for idx,labels in zip(fidx,flabels):
        #        for label in labels.rstrip().split(" "):
        #            try:
        #                self.labels[label].append(idx.rstrip())
        #            except AttributeError:
        #                self.labels[label].append(idx)
        #else:
        #    logging.warning("labels not found, continue without labels")

    def __iter__(self):
        Data = namedtuple('paper', 'idx words citations')

        fidx   = open(self.path+"/idxs.txt")
        fwords = open(self.path+"/texts.txt")
        flinks = open(self.path+"/links.txt")

        i = 0
        for idx,words,link in zip(fidx,fwords,flinks):
            i += 1
            if self.debug and i > 10000:
                return
            yield Data(
                        idx[:-1] if isinstance(idx,str) else idx,
                        words[:-1].split(" "),
                        link[:-1].split(" ") if len(link) > 1 else [],
                    )

    def clear(self):
        del self.w2v_counter
        del self.d2v_counter
        del self.n2v_counter

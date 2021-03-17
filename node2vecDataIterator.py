import logging
import networkx as nx
from collections import defaultdict, namedtuple
import os
import numpy as np
try:
    import ujson as json
except ImportError:
    import json

class DataIterator():
    def __init__(self,filename,walking_path=None,length_per_node = 100,p=0.25,q=0.25,iter=5, undirected=True,cache = None,delimiter=","):
        self.length_per_node = length_per_node
        self.degree = defaultdict(int)
        self.iter = iter
        self.alg = 'node2vec'


        cites = defaultdict(list)
        logging.info("load network for node2vec")
        with open(filename) as f:
            for line in f:
                source,target = line.rstrip().split(delimiter)
                self.degree[source] += 1
                if undirected:
                    self.degree[target] += 1
                cites[source].append(target)

        self.network = cites


        nx_G = nx.Graph()
        for source,targetList in cites.items():
            for target in targetList:
                nx_G.add_edge(source,target,weight=1)
                
        if undirected:
            nx_G = nx_G.to_undirected()

        self.node2vec = Node2Vec(nx_G, not undirected, p, q)


        if cache is not None and os.path.exists(cache+".alias_nodes")and os.path.exists(cache+".alias_edges"):
            logging.info("found cache for node2vec, loading")
            with open(cache+".alias_nodes",'r') as f:
                self.node2vec.alias_nodes = {}
                for line in f:
                    key,value = json.loads(line.rstrip())
                    self.node2vec.alias_nodes[key] = value
            with open(cache+".alias_edges",'r') as f:
                self.node2vec.alias_edges= {}
                for line in f:
                    key,value = json.loads(line.rstrip())
                    self.node2vec.alias_edges[key] = value
        else:
            logging.info("building cache for node2vec")
            logging.info("preprocessing transition probabilities")
            self.node2vec.preprocess_transition_probs()
            logging.info("done")
            if cache is not None:
                with open(cache+".alias_nodes",'w') as f:
                    for key,value in self.node2vec.alias_nodes.items():
                        f.write(json.dumps((key,value)))
                        f.write("\n")
                with open(cache+".alias_edges",'w') as f:
                    for key,value in self.node2vec.alias_edges.items():
                        f.write(json.dumps((key,value)))
                        f.write("\n")

        self.walking_path = walking_path
        if self.walking_path:
            self.walk(walking_path)



    def walk(self,walking_path):
        with open(walking_path,'w') as walking_path:
            for i in range(self.iter):
                for index,source in enumerate(self.node2vec.G.nodes()):
                    if index % 1000 == 0:
                        logging.info("walking %d/%d (%d)" % (i+1,self.iter,index))
                    path = [x for x in self.node2vec.node2vec_walk(walk_length=self.length_per_node, start_node=source)]
                    walking_path.write(" ".join(path))
                    walking_path.write("\n")


    def __iter__(self):
        if self.walking_path is None:
            raise ValueError("You must perform walking before training")
        if not os.path.exists(self.walking_path):
            logging.error("%s file not found!" % self.walking_path)
            self.walk(self.walking_path)
        data = namedtuple('node2vec', 'idx words citations')
        counter = 0
        with open(self.walking_path) as f:
            for idx,line in enumerate(f):
                words = line.split()
                if len(words) == 0:
                    continue
                if idx > 0 and idx % 1000000 == 0:
                    logging.info("%s lines loaded" % idx)
                counter += len(words)
                if counter >= 1e9:
                    break
                yield data(
                        idx,
                        words,
                        []
                        )


class Node2Vec():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.nodes = self.G.nodes
        self.Graph = self.G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    key = "%s$$%s" % (prev,cur)
                    if key not in alias_edges:
                        key = "%s$$%s" % (cur,prev)
                    next = cur_nbrs[alias_draw(alias_edges[key][0],
                        alias_edges[key][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        logging.info('Walk iteration:')
        for walk_iter in range(num_walks):
            logging.info(str(walk_iter+1)+'/'+str(num_walks))
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)
                                                        
    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        logging.info("preprocess nodes")
        no_nodes = G.number_of_nodes()
        for index,node in enumerate(G.nodes()):
            if index % max(10000,(no_nodes // 10000)) == 0:
                logging.info("%d%% nodes preprocessed" % (index/no_nodes*100))
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        logging.info("preprocess edges")
        no_edges = G.number_of_edges()
        if is_directed:
            for index,edge in enumerate(G.edges()):
                if index % max(10000,(no_edges // 10000)) == 0:
                    logging.info("%d%% edges preprocessed" % (index/no_edges*100))
                key = "%s$$%s" % edge
                alias_edges[key] = self.get_alias_edge(edge[0], edge[1])
        else:
            for index,edge in enumerate(G.edges()):
                if index % max(10000,(no_edges// 100)) == 0:
                    logging.info("%d%% edges  preprocessed" % (index/no_edges*100))
                key = "%s$$%s" % edge
                key2 = "%s$$%s" % (edge[1],edge[0])
                alias_edges[key] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[key2] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return
def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J.tolist(), q.tolist()

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


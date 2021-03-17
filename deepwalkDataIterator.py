import logging
from collections import defaultdict, namedtuple
import os
import pickle
import numpy as np
class DataIterator():
    def __init__(self,filename,walking_path = None, length_per_node = 100,iter=5, undirected=True, delimiter=","):
        self.length_per_node = length_per_node
        self.degree = defaultdict(int)
        self.iter = iter
        self.alg = 'DeepWalk'

        cites = defaultdict(list)
        with open(filename) as f:
            for line in f:
                source,target = line.rstrip().split(delimiter)
                cites[source].append(target)
                self.degree[source] += 1
                if undirected:
                    self.degree[target] += 1
                    cites[target].append(source)


        self.network = cites
        self.walking_path = walking_path
        if self.walking_path:
            self.walk(walking_path)

    def walk(self,walking_path):
        self.walking_path = walking_path
        with open(walking_path,'w') as walking_path:
            for i in range(self.iter):
                logging.info("walking %d/%d" % (i+1,self.iter))
                for source in self.network:
                    path = [source]
                    while len(path) < self.length_per_node:
                        if path[-1] not in self.network:
                            path.append(source)
                            continue
                        n = self.network[path[-1]][np.random.randint(len(self.network[path[-1]]))] if len(self.network[path[-1]]) > 0 else source
                        #n = np.random.choice(self.network[path[-1]]) if len(self.network[path[-1]]) > 0 else source
                        path.append(n)
                    walking_path.write(" ".join(path))
                    walking_path.write("\n")

    def __iter__(self):
        if self.walking_path is None:
            raise ValueError("You must perform walking before training")
        if not os.path.exists(self.walking_path):
            logging.error("%s file not found!" % self.walking_path)
            self.walk(self.walking_path)
        data = namedtuple('DeepWalk', 'idx words citations')
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



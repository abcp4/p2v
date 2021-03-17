import logging
from collections import defaultdict, namedtuple
import os
import pickle
class DataIterator():
    def __init__(self,filename,undirected=True,delimiter = ","):

        logging.info("loading network")
        self.alg = 'LINE'
        self.network = defaultdict(list)
        with open(filename) as f:
            for line in f:
                line = line.rstrip().split(delimiter)
                self.network[line[0]].append(line[1])
                if not undirected:
                    self.network[line[1]].append(line[0])


        logging.info("loading labels")
        logging.info("done... %s papers loaded" % (len(self.network)))

    def __iter__(self):
        data = namedtuple('LINE', 'idx words citations')
        for source,targets in self.network.items():
            yield(
                    data(
                        source,
                        [],
                        targets
                        )
                    )


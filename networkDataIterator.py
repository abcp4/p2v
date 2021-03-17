import logging
from collections import defaultdict, namedtuple
import os
import pickle
class DataIterator():
    def __init__(self,filename,name=None,delimiter = "\t"):
        self.name = name if name else "BlogCatalog"

        logging.info("loading network")
        self.network = defaultdict(list)
        with open(filename) as f:
            for line in f:
                line = line.rstrip().split(delimiter)
                if len(line) != 2:
                    continue
                self.network[line[0]].append(line[1])


        logging.info("loading labels")
        logging.info("done... %s papers loaded" % (len(self.network)))

    def __iter__(self):
        data = namedtuple('network', 'idx words citations')
        for source,targets in self.network.items():
            yield(
                    data(
                        source,
                        [],
                        targets
                        )
                    )

    def __str__(self):
        return self.name

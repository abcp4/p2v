from collections import defaultdict, namedtuple
import logging
import gzip
import cython

class DataIterator():
    def __init__(self,filename,lower = False):
        self.filename = filename
        self.lower = lower

    def __str__(self):
        return "sentence"

    def __iter__(self):
        data = namedtuple('sentence', 'idx words citations')
        counter = 0
        with open(self.filename) as f:
            for idx,line in enumerate(f):
                if self.lower:
                    words = line.lower().split()
                else:
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



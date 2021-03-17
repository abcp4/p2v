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
    def __init__(self,data_path='./',ngram=None):
        self.path = data_path
        self.ngram = ngram

    def __iter__(self):
        Data = namedtuple('paper', 'idx words citations')

        if os.path.exists(self.path+"/idxs.txt"):
            fidx   = open(self.path+"/idxs.txt")
        else:
            fidx = range(file_len(self.path+"/texts.txt"))

        fwords = open(self.path+"/texts.txt")

        if self.ngram is None:
            for idx,words in zip(fidx,fwords):
                yield Data(
                            idx[:-1] if isinstance(idx,str) else idx,
                            words[:-1].split(" "),
                            []
                        )
        else:
            for idx,words in zip(fidx,fwords):
                yield Data(
                            idx[:-1] if isinstance(idx,str) else idx,
                            ngrams(words[:-1].split(" "),self.ngram),
                            []
                        )

def ngrams(token, n):
    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    return [" ".join(x) for x in zip(*[token[i:] for i in range(n)])]

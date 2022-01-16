#!/usr/bin/env python
# %%

import os
import random

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

class Dataset:
    def __init__(self, bytes, seed=None):
        self.bytes = bytes
        self.rng = random.Random(seed)
    
    def __len__(self):
        return len(self.bytes)

    @classmethod
    def from_file(cls, path):
        with open(path, 'rb') as f:
            return cls(bytes=f.read())

    def sample(self, n):
        ''' Sample a slice of length n '''
        assert n <= len(self), f'n={n} > len(self)={len(self)}'
        start = self.rng.randint(0, len(self.bytes) - n)
        return self.bytes[start:start+n]

# tiny = Dataset(bytes=b'abc')
# tiny.sample(4)

kjv = Dataset.from_file(os.path.join(DATA_PATH, 'kjv.txt'))
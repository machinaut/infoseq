#!/usr/bin/env python

# %%
import random

class Tree:
    def __init__(self, token, children=None, seed=None):
        self.token = token
        self.children = children or [None] * 256
        self.rng = random.Random(seed)

    def add(self, token, text):
        b = text[0]
        if self.children[b] is None:
            self.children[b] = Tree(token)
        if len(text) > 1:
            self.children[b].add(token, text[1:])
    
    def encode_single(self, text, compression=.9):
        # Leaf node
        if not len(text):
            return self.token, b''
        t = text[0]
        assert 0 <= t < 256, f't={t}'
        # Stochastically choose to leave the tree early
        if self.rng.random() > compression:
            return self.token, text
        # Otherwise, recurse to children if possible
        if self.children[t] is None:
            return self.token, text
        return self.children[t].encode_single(text[1:], compression=compression)

    def encode(self, text, compression=.9):
        tokens = []
        while len(text):
            token, text = self.encode_single(text, compression=compression)
            tokens.append(token)
        return tokens

    def check(self, text):
        if not len(text):
            return True
        t = text[0]
        assert 0 <= t < 256, f't={t}'
        if self.children[t] is None:
            return False
        return self.children[t].check(text[1:])

    @property
    def encode_map(self):
        encode_map = {b'': self.token}
        for t, child in enumerate(self.children):
            if child is not None:
                for text, token in child.encode_map.items():
                    encode_map[bytes([t]) + text] = token
        return encode_map

    @property
    def decode_map(self):
        return {v: k for k, v in self.encode_map.items()}

tree = Tree(0)
tree.add(1, b'a')
tree.add(2, b'b')
tree.add(3, b'c')
tree.add(4, b'ab')
tree.encode(b'abc')
# tree.encode_map
# tree.decode_map

# %%

class Tokenization:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.tree = Tree(0)
        self.decode_map = {0: b''}
        # Start with single bytes
        for i in range(256):
            self.add(bytes([i]))

    def encode(self, text, compression=.9):
        ''' Search forwards from shortest possible encoding '''
        return self.tree.encode(text, compression=compression)

    def decode(self, tokens):
        return b''.join(self.decode_map[t] for t in tokens)

    def add(self, text):
        ''' Add bytes as a token, and every left partial '''
        if len(text) > 1:
            self.add(text[:-1])
        if not self.tree.check(text):
            token = len(self.decode_map)
            assert token not in self.decode_map, f'token={token}'
            self.tree.add(token, text)
            self.decode_map[token] = text

tok = Tokenization()
tok.decode(tok.encode(b'abcd'))

# %%
def create_bpe(dataset)
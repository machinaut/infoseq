#!/usr/bin/env python
# %% trie data structure
from collections import defaultdict


class Trie:
    def __init__(self, parent=None, byte=None):
        assert parent is None or isinstance(parent, Trie), f'{parent}'
        assert byte is None or isinstance(byte, int), f'{byte}'
        self.children = [None] * 256
        self.parent = parent
        self.byte = byte
        self.token = None

    @property
    def is_root(self):
        return self.parent is None

    @classmethod
    def from_decode(cls, decode):
        root = cls()
        for k, v in decode.items():
            root.add(v, k)

    @classmethod
    def from_encode(cls, encode):
        return cls.from_decode({v: k for k, v in encode.items()})

    def add(self, value, token):
        assert isinstance(value, bytes), f'{value}'
        assert isinstance(token, int), f'{token}'
        byte, value = value[0], value[1:]
        if not value:
            self.get_child(byte).token = token
        else:
            self.get_child(byte).add(value, token)

    def get_child(self, byte):
        assert isinstance(byte, int), f'{byte}'
        if self.children[byte] is None:
            self.children[byte] = Trie(self, byte)
        return self.children[byte]

    def __repr__(self, prefix=''):
        children = [c.__repr__(prefix + '  ') for c in self.children if c]
        byte = str(bytes([self.byte])) if self.byte is not None else ' '
        token = self.token if self.token is not None else ''
        first = f'{prefix}* {byte[1:]}:{token}'
        return '\n'.join([first] + children)


root = Trie()
root.add(b'a', 0)
root.add(b'b', 1)
root.add(b'c', 2)
root.add(b'd', 3)
root.add(b'aa', 4)
root.add(b'ab', 5)
root.add(b'aaab', 6)
print(repr(root))

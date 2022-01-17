#!/usr/bin/env python
import json
import random


class Tokenization:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.tree = {}
        self.encode_map = {b'': 0}
        self.decode_map = {0: b''}

    def __len__(self):
        return len(self.decode_map)

    def __contains__(self, item):
        return item in self.encode_map or item in self.decode_map

    def add(self, text: bytes):
        ''' Add a new code to the tree, returning the token '''
        assert text, repr(text)
        assert text not in self.encode_map, f'{repr(text)} already in encode_map'
        assert text[:-1] in self.encode_map, f'{repr(text[:-1])} not in encode_map'
        # Add new token to our maps
        token = len(self.encode_map)
        assert token not in self.decode_map, f'{token} already in decode_map'
        self.encode_map[text] = token
        self.decode_map[token] = text
        # Traverse the tree to add the node
        current = self.tree
        while text:
            t, *text = text
            if t not in current:
                current[t] = {}
            current = current[t]
        return token

    def encode_step(self, text: bytes, compression: float = 0.9):
        ''' Encode a single step, returning the next token and remaining text '''
        assert 0.0 <= compression <= 1.0, repr(compression)
        # Traverse the tree
        code = []
        current = self.tree
        while text and text[0] in current and self.rng.random() < compression:
            code.append(text[0])
            current = current[text[0]]
            text = text[1:]
        # Return the token and remaining text
        return self.encode_map[bytes(code)], text

    def encode(self, text: bytes, compression: float = 0.9):
        ''' Encode a string, returning a list of tokens '''
        assert 0.0 <= compression <= 1.0, repr(compression)
        # Encode the text step by step
        tokens = []
        while len(text):
            token, text = self.encode_step(text, compression=compression)
            tokens.append(token)
        return tokens

    def decode(self, tokens: list):
        ''' Decode a list of tokens, returning the original text '''
        return b''.join(self.decode_map[t] for t in tokens)

    def __eq__(self, obj: object) -> bool:
        ''' Test that another tokenization equals this one '''
        if not isinstance(obj, Tokenization):
            return False
        return dict(self) == dict(obj)

    def __iter__(self):
        ''' Only iterate over decode map, enough to reconstruct the tokenization '''
        for token, code in sorted(self.decode_map.items()):
            yield token, code

    def to_json(self):
        ''' Return a JSON representation of the tokenization '''
        d = {str(k): list(v) for k, v in dict(self).items()}
        return json.dumps(d)

    @classmethod
    def from_json(cls, s):
        ''' Reconstruct a tokenization from a JSON string '''
        d = {int(k): bytes(v) for k, v in json.loads(s).items()}
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict):
        ''' Reconstruct a tokenization from a dict of token -> code '''
        # Check that the dict keys are all ints
        assert all(isinstance(k, int) for k in d.keys()), repr(d)
        # Check that the dict is not missing keys
        assert set(d.keys()) == set(range(len(d))), repr(d)
        # Check that the values are all bytes
        assert all(isinstance(v, bytes) for v in d.values()), repr(d)
        # Check that the zero token (which we start with) is correct
        assert d[0] == b'', f'missing zero token: {d}'
        # Build the tokenization by adding all the codes in order
        tok = cls()
        for token, code in sorted(d.items()):
            if token > 0:  # skip the zero token
                tok.add(code)
        # Check our tokenization is correct
        assert dict(tok) == d, f'{dict(tok)} != {d}'
        return tok

    @classmethod
    def basic(cls, seed=None):
        ''' Return a tokenization with just single bytes '''
        tok = cls(seed=seed)
        # Add all single-byte tokens
        for b in range(256):
            tok.add(bytes([b]))
        return tok
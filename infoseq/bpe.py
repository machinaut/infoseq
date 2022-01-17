#!/usr/bin/env python
import random
from collections import defaultdict
from infoseq.token import Tokenization


def bpe_from_text(text, max_tokens=1000, compression=0.9, seq_len=300, num_seq=300, seed=None):
    """Create a Byte Pair Encoding (BPE) tokenization"""
    tok = Tokenization(seed=seed)
    # Add all unique bytes present in the text
    for b in set(text):
        tok.add(bytes([b]))
    # Add pairs until we reach max_tokens
    while len(tok) < max_tokens:
        # Get pairs until we get something new
        for code in get_pairs(text, tok, compression=compression, seq_len=seq_len, num_seq=num_seq, seed=seed):
            if code not in tok:
                break
        else:
            assert False, "Could not find new code"
        # Add the smallest sub-code which is new
        for i in range(1, len(code) + 1):
            if code[:i] not in tok:
                tok.add(code[:i])
                break
        else:
            assert False, f"{code} ({len(code)}) : {code in tok}"
    return tok


def get_pairs(text: bytes, tok: Tokenization, compression=0.9, seq_len=300, num_seq=300, seed=None):
    """Get possible token code pairs"""
    assert len(text) >= seq_len, f"{len(text)} < {seq_len}"
    rng = random.Random(seed)
    # count instances of token code pairs
    pairs = defaultdict(int)
    for _ in range(num_seq):
        start = rng.randint(0, len(text) - seq_len)
        tokens = tok.encode(text[start: start + seq_len],
                            compression=compression)
        for i in range(len(tokens) - 1):
            # directly combine two tokens into a single code
            code = tok.decode_map[tokens[i]] + tok.decode_map[tokens[i + 1]]
            pairs[code] += 1
    # return token code pairs as codes in order
    return sorted(pairs, key=pairs.get, reverse=True)

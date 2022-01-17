#!/usr/bin/env python

import unittest
from infoseq.token import Tokenization
from infoseq.bpe import get_pairs, bpe_from_text


class TestBPE(unittest.TestCase):
    def test_pair(self):
        tok = Tokenization.basic()
        text = b'ab'
        pairs = get_pairs(text, tok, 1.0, len(text), 1, 0)
        self.assertEqual(pairs[0], b'ab')

    def test_pairs(self):
        # https://en.wikipedia.org/wiki/Byte_pair_encoding
        text = b'aaabdaaabac'
        tok = Tokenization.basic()
        pairs = get_pairs(text, tok, 1.0, len(text), 1, 0)
        self.assertEqual(pairs[0], b'aa')
        tok.add(b'aa')
        pairs = get_pairs(text, tok, 1.0, len(text), 1, 0)
        self.assertEqual(pairs[0], b'aaa')
        tok.add(b'aaa')
        pairs = get_pairs(text, tok, 1.0, len(text), 1, 0)
        self.assertEqual(pairs[0], b'aaab')

    def test_bpe(self):
        text = b'aaabdaaabac'
        tok = bpe_from_text(text, max_tokens=257 + 3,
                            compression=1.0, seq_len=len(text), num_seq=1, seed=0)
        self.assertIn(b'aa', tok)
        self.assertIn(b'aaa', tok)
        self.assertIn(b'aaab', tok)


if __name__ == '__main__':
    unittest.main()

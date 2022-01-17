#!/usr/bin/env python
import json
import random
import unittest

from infoseq.token import Tokenization


class TestTokenization(unittest.TestCase):
    def test_recode(self):
        rng = random.Random(0)
        t = Tokenization(0)
        self.assertEqual(t.add(b'a'), 1)
        self.assertEqual(t.add(b'b'), 2)
        self.assertEqual(t.add(b'c'), 3)
        self.assertEqual(len(t), 4)  # includes empty string
        for _ in range(100):
            text = bytes(rng.choices(b'abc', k=rng.randint(1, 10)))
            self.assertEqual(text, t.decode(t.encode(text)))

    def test_contains(self):
        ''' Check the 'in' operator works for codes and tokens '''
        tok = Tokenization(0)
        self.assertEqual(tok.add(b'a'), 1)
        self.assertEqual(tok.add(b'b'), 2)
        self.assertEqual(tok.add(b'ab'), 3)
        self.assertIn(b'a', tok)
        self.assertIn(b'b', tok)
        self.assertIn(b'ab', tok)
        self.assertIn(0, tok)
        self.assertIn(1, tok)
        self.assertIn(2, tok)
        self.assertIn(3, tok)
        self.assertNotIn(b'c', tok)
        self.assertNotIn(b'abc', tok)
        self.assertNotIn(4, tok)

    def test_basic(self):
        ''' Test the basic tokenization '''
        rng = random.Random(0)
        t = Tokenization.basic()
        for _ in range(100):
            text = bytes(rng.choices(b'abc', k=rng.randint(1, 10)))
            self.assertEqual(text, t.decode(t.encode(text)))

    def test_equals(self):
        ''' Test equality '''
        tok0 = Tokenization(0)
        tok1 = Tokenization(1)
        for code in [b'a', b'b', b'c', b'ab']:
            tok0.add(code)
            tok1.add(code)
        self.assertEqual(tok0, tok1)

    def test_deterministic(self):
        ''' Test a simple deterministic case '''
        tok = Tokenization(0)
        self.assertEqual(tok.add(b'a'), 1)
        self.assertEqual(tok.add(b'b'), 2)
        self.assertEqual(tok.add(b'c'), 3)
        self.assertEqual(tok.add(b'ab'), 4)
        self.assertEqual([4, 3], tok.encode(b'abc', compression=1.0))

    def test_stochastic(self):
        ''' Test that we get output with different encodings '''
        tok = Tokenization(0)
        self.assertEqual(tok.add(b'a'), 1)
        self.assertEqual(tok.add(b'b'), 2)
        self.assertEqual(tok.add(b'c'), 3)
        self.assertEqual(tok.add(b'ab'), 4)
        encodings = set(tuple(tok.encode(b'abc', compression=0.7))
                        for _ in range(100))
        expected = [
            (4, 3),
            (0, 4, 3),
            (4, 0, 3),
            (1, 2, 3),
            (0, 1, 2, 3),
            (1, 0, 2, 3),
            (1, 2, 0, 3),
        ]
        for ex in expected:
            self.assertIn(ex, encodings)

    def test_compression(self):
        ''' Test that the compression ratio is approximately correct '''
        N = 100
        tok = Tokenization(0)
        self.assertEqual(tok.add(b'a'), 1)
        encodings = [tuple(tok.encode(b'a', compression=0.5))
                     for _ in range(N)]
        # at least 40% without zero pad
        self.assertGreaterEqual(encodings.count((1,)), N * .4)
        # at least 20% with zero pad
        self.assertGreaterEqual(encodings.count((0, 1)), N * .2)
        # at least 10% with double zero pad
        self.assertGreaterEqual(encodings.count((0, 0, 1)), N * .1)
        # Make sure we never end with empty string
        self.assertNotIn((1, 0), encodings)

    def test_iter(self):
        ''' Test iterating over a tokenization '''
        tok = Tokenization(0)
        self.assertEqual(tok.add(b'a'), 1)
        self.assertEqual(tok.add(b'b'), 2)
        self.assertEqual(tok.add(b'c'), 3)
        self.assertEqual(tok.add(b'ab'), 4)
        expected = [b'', b'a', b'b', b'c', b'ab']
        for i, (token, code) in enumerate(tok):
            self.assertEqual(token, i)
            self.assertEqual(code, expected[i])
        self.assertEqual(len(list(tok)), 5)

    def test_dict(self):
        ''' Test converting tokenization to a dictionary '''
        tok = Tokenization(0)
        self.assertEqual(tok.add(b'a'), 1)
        self.assertEqual(tok.add(b'b'), 2)
        self.assertEqual(tok.add(b'c'), 3)
        self.assertEqual(tok.add(b'ab'), 4)
        expected = [b'', b'a', b'b', b'c', b'ab']
        d = dict(enumerate(expected))
        self.assertEqual(dict(tok), d)

    def test_json(self):
        ''' Test converting to/from JSON '''
        tok = Tokenization(0)
        self.assertEqual(tok.add(b'a'), 1)
        self.assertEqual(tok.add(b'b'), 2)
        self.assertEqual(tok.add(b'c'), 3)
        self.assertEqual(tok.add(b'ab'), 4)
        expected = {'0': [], '1': list(b'a'), '2': list(
            b'b'), '3': list(b'c'), '4': list(b'ab')}
        self.assertEqual(tok.to_json(), json.dumps(expected))
        self.assertEqual(Tokenization.from_json(json.dumps(expected)), tok)


if __name__ == '__main__':
    unittest.main()

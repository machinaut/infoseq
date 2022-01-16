#!/usr/bin/env python
# %%
import unittest
import random
from infoseq.token import Tree


class TestTree(unittest.TestCase):
    def test_recode(self):
        rng = random.Random(0)
        t = Tree(0)
        t.add(1, b'a')
        t.add(2, b'b')
        t.add(3, b'c')
        for _ in range(10):
            text = bytes(rng.choices(b'abc', k=rng.randint(1, 10)))
            self.assertEqual(text, t.decode(t.encode(text)))

tt = TestTree()
tt.test_recode()



# %%
print('hi')

# %%
if __name__ == '__main__':
    unittest.main()
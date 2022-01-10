#! /usr/bin/env python3

import numpy as np

class Lazy_Iterable_Wrapper:
    """
    Apply a lambda to each element from an iterable. Requires __len__ to be implemented
    """
    def __init__(self, iterable, lam) -> None:
        self.iterable = iterable
        self.lam = lam
        self.debug = False

    def __iter__(self):
        self.debug = True
        self.it = iter(self.iterable)
        return self

    def __next__(self):
        return self.lam(next(self.it))

    def __len__(self):
        return len(self.iterable)
    

if __name__ == "__main__":
    import unittest
    import random

    LEN_SEQUENCE = 100000
    LAM = lambda x: x*x

    class test_File_As_Windowed_Sequence(unittest.TestCase):
        def test(self):
            l = range(1000)

            lam = lambda x: x*x

            self.assertEqual(
                list(map(lam, l)),
                list(Lazy_Iterable_Wrapper(l, lam))
            )


    unittest.main()

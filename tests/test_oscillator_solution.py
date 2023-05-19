from src.Oscillators.oscillator_solution import OscillatorSolution
from src.Oscillators import util
from src.Oscillators import theta, k_d, t

import sympy as sp
import unittest

IGNORE_TEST = False
IS_PLOT = False

class TestOscillatorSolution(unittest.TestCase):

    def setUp(self):
        self.soln = OscillatorSolution()

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.soln.A_mat.shape, (2, 2))

    def testGetSolution(self):
        if IGNORE_TEST:
            return
        self.soln.getSolution(is_check=False)
        self.assertEqual(self.soln.x_vec.shape, (2, 1))

    def testFindSinusoidCoefficients(self):
        if IGNORE_TEST:
            return
        expression = sp.cos(t*theta)*k_d + sp.sin(t*theta)*k_d + 1
        dct = util.findSinusoidCoefficients(expression)
        for key in ["a", "b", "c"]:
            self.assertTrue(key in dct.keys())

if __name__ == "__main__":
    unittest.main()
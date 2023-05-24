from src.Oscillators.solver import Solver
from src.Oscillators import util
from src.Oscillators import theta, k_d, t

import lmfit
import os
import pandas as pd
import sympy as sp
import unittest

IGNORE_TEST = False
IS_PLOT = False
TEST_DIR = os.path.dirname(os.path.abspath(__file__)) # This directory
TEST_FILE = os.path.join(TEST_DIR, "test_oscillator_solution.pdf")
REMOVE_FILES = [TEST_FILE] 

class TestOscillatorSolution(unittest.TestCase):

    def setUp(self):
        self.soln = Solver()
        self.remove()

    def tearDown(self):
        self.remove()

    def remove(self):
        for path in REMOVE_FILES:
            if os.path.exists(path):
                os.remove(path)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.soln.A_mat.shape, (2, 2))

    def testSolve(self):
        if IGNORE_TEST:
            return
        self.soln.solve(is_check=False)
        self.assertEqual(self.soln.x_vec.shape, (2, 1))

    def testFindSinusoidCoefficients(self):
        if IGNORE_TEST:
            return
        expression = sp.cos(t*theta)*k_d + sp.sin(t*theta)*k_d + 1
        dct = util.findSinusoidCoefficients(expression)
        for key in ["a", "b", "c"]:
            self.assertTrue(key in dct.keys())

    def testSimulate(self):
        if IGNORE_TEST:
            return
        self.soln.solve(is_check=False)
        df = self.soln.simulate(is_plot=IS_PLOT, output_path=TEST_FILE, title="testSimulate")
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertGreater(len(df), 0)
        self.assertTrue(all([col in df.columns for col in ["S1", "S2"]])) 

    def testCalculateResiduals(self):
        if IGNORE_TEST:
            return

if __name__ == "__main__":
    unittest.main()
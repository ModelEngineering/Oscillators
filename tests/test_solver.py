from src.Oscillators.solver import Solver
from src.Oscillators import util
from src.Oscillators import theta, k_d, t
import src.Oscillators.constants as cn

import numpy as np
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
        self.soln.solve(is_check=False, is_simplify=False)
        self.assertEqual(self.soln.x_vec.shape, (2, 1))

    def testSolve2(self):
        if IGNORE_TEST:
            return
        old_soln = self.soln.deprecatedSolve(is_check=False, is_simplify=False)
        new_soln = self.soln.solve(is_check=False, is_simplify=False)
        for _ in range(10):
            parameter_dct = util.makeRandomParameterDct(max_val=100, is_t=True)
            symbol_dct = util.makeSymbolDct(old_soln, parameter_dct)
            old_val = sp.N(old_soln.subs(symbol_dct))
            new_val = sp.N(new_soln.subs(symbol_dct))
            is_0 = np.isclose(float(old_val[0]), float(new_val[0]))
            is_1 = np.isclose(float(old_val[1]), float(new_val[1]))
            self.assertTrue(is_0 and is_1)

    def testCalculateOscillationCharacteristics(self):
        if IGNORE_TEST:
            return
        result = self.soln.calculateOscillationCharacteristics(is_check=False, is_simplify=False)
        self.assertTrue(isinstance(result, dict))
        self.assertEqual(len(result), 4)

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

    def testGetOscillatorCharacteristics(self): 
        if IGNORE_TEST:
            return
        self.soln.solve(is_check=False)
        oc1, oc2 = self.soln.getOscillatorCharacteristics(dct=cn.PARAM_DCT)
        self.assertEqual(oc1.theta, oc2.theta)

    def testPlotFit(self):
        if IGNORE_TEST:
            return
        self.soln.solve(is_check=False)
        self.soln.plotFit(is_plot=IS_PLOT, output_path=TEST_FILE, title="testPlotFit")
        #
        dct = {}
        dct["k2"] = [11.3560414, 5.97, 9.78, 16.81]
        dct["k_d"] = [2.2014717, 4.18, 10, 23, 5.94]
        dct["k4"] = [118.818599, 372.57, 119.92, 777.43]
        dct["k6"] = [129.83, 592.03, 169.95, 993.03]
        dct["x1_0"] = [5.0, 57.33, 5.0, 40.64]
        dct["x2_0"] = [7.66, 10.0, 2.038, 10.0]
        #
        param_dct = {n: v[0] for n, v in dct.items()}
        self.soln.plotFit(is_plot=IS_PLOT, param_dct=param_dct, output_path=TEST_FILE, title="testPlotFit1")

    def testPlotManyFits(self):
        if IGNORE_TEST:
            return
        self.soln.solve(is_check=False)
        self.soln.plotManyFits(is_plot=IS_PLOT, output_path=TEST_FILE)

if __name__ == "__main__":
    unittest.main()
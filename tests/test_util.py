'''Tests for util.py'''

from src.Oscillators import k_d, t, theta, k2
from src.Oscillators.constants import PARAM_DCT 
from src.Oscillators import util
import src.Oscillators.constants as cn
from src.Oscillators.util import TIMES

import os
import pandas as pd
import sympy as sp
import unittest

IGNORE_TEST = False
IS_PLOT = False
TEST_DIR = os.path.dirname(os.path.abspath(__file__)) # This directory
TEST_FILE = os.path.join(TEST_DIR, "test_util.pdf")
REMOVE_FILES = [TEST_FILE] 


class TestUtil(unittest.TestCase):

    def setUp(self):
        self.remove()

    def tearDown(self):
        self.remove()

    def testPlotDF(self):
        if IGNORE_TEST:
            return
        data = range(len(TIMES))
        df = pd.DataFrame({"A": data, "B": data}, index=TIMES)
        util.plotDF(df, is_plot=IS_PLOT, output_path=TEST_FILE)
        self.assertTrue(os.path.isfile(TEST_FILE))

    def testMakeTimes(self):
        if IGNORE_TEST:
            return
        density = 7
        times = util.makeTimes(0, 5, density)
        self.assertEqual(len(times), 5*density)
        self.assertEqual(times[0], 0)
        self.assertEqual(times[-1], 5) 

    def remove(self):
        for path in REMOVE_FILES:
            if os.path.exists(path):
                os.remove(path)

    def testSimulateLinearSystem(self):
        if IGNORE_TEST:
            return
        df = util.simulateLinearSystem(is_plot=IS_PLOT, output_path=TEST_FILE)
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertGreater(len(df), 0)
        self.assertTrue(all([col in df.columns for col in ["S1", "S2"]]))

    def testSimulateExpressionVector(self):
        if IGNORE_TEST:
            return
        vec = sp.Matrix([t, t**2])
        _ = util.simulateExpressionVector(vec, {}, is_plot=IS_PLOT)

    def testSimulateRR(self):
        if IGNORE_TEST:
            return
        df = util.simulateRR(is_plot=IS_PLOT, output_path=TEST_FILE)
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertGreater(len(df), 0)

    def testMakeSymbolDct(self):
        if IGNORE_TEST:
            return
        dct = util.makeSymbolDct(theta*k_d, PARAM_DCT, exclude_names=["t"])
        self.assertEqual(dct[theta], PARAM_DCT["theta"])

    def testMakePolynomialCoefficients(self):
        if IGNORE_TEST:
            return
        expression = 3*sp.cos(t*theta)*sp.cos(theta*t) + 2*sp.cos(2*t*theta-t*theta) + 1
        dct = util.makePolynomialCoefficients(expression, sp.cos(t*theta))
        trues = [dct[n] == n+1 for n in range(2)]
        self.assertTrue(all(trues))

    def testGetSubstitutedExpression(self):
        if IGNORE_TEST:
            return
        dct = {"theta": 2, "k2":4.0}
        expression = 2*theta*k2
        result = util.getSubstitutedExpression(expression, dct)
        self.assertEqual(result, 16)
        self.assertTrue(isinstance(result, float))
        #
        expression = 2*theta*k2*k_d
        result = util.getSubstitutedExpression(expression, dct)
        self.assertTrue("sympy" in str(type(result)))

    def testMakeUniformRandomParameterDct(self):
        if IGNORE_TEST:
            return
        parameters = list(cn.INDEPENDENT_PARAMETERS)
        result_dct = util.makeUniformRandomParameterDct(parameters=parameters, is_calculate_dependent_parameters=True)
        self.assertTrue(isinstance(result_dct, dict))
        trues = [isinstance(result_dct[k], float) for k in result_dct.keys()]
        self.assertTrue(all(trues))
        self.assertTrue(cn.C_K3 in result_dct.keys())
        self.assertTrue(cn.C_K5 in result_dct.keys())
        #
        result_dct = util.makeUniformRandomParameterDct(parameters=parameters, is_calculate_dependent_parameters=False)
        self.assertFalse(cn.C_K3 in result_dct.keys())
        self.assertFalse(cn.C_K5 in result_dct.keys())

    def testMakeNormalRandomParameterDct(self):
        if IGNORE_TEST:
            return
        mean_dct = {k: 1 for k in cn.ALL_PARAMETERS}
        result = util.makeNormalRandomParameterDct(cv=0.1, means=mean_dct)
        self.assertTrue(isinstance(result, dict))
        trues = [isinstance(result[k], float) for k in result.keys()]
        self.assertTrue(all(trues))

if __name__ == "__main__":
    unittest.main()
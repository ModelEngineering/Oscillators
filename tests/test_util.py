'''Tests for util.py'''

from src.Oscillators import k_d, t, theta
from src.Oscillators.model import MODEL, PARAM_DCT 
from src.Oscillators import util
from src.Oscillators.util import TIMES, MODEL

import sympy as sp
import tellurium as te
import unittest

IGNORE_TEST = False
IS_PLOT = True

class TestUtil(unittest.TestCase):

    def testMakeTimes(self):
        if IGNORE_TEST:
            return
        density = 7
        times = util.makeTimes(0, 5, density)
        self.assertEqual(len(times), 5*density)
        self.assertEqual(times[0], 0)
        self.assertEqual(times[-1], 5) 

    def testSimulateLinearSystem(self):
        if IGNORE_TEST:
            return
        util.simulateLinearSystem(is_plot=IS_PLOT)

    def testSimulateExpressionVector(self):
        vec = sp.Matrix([t, t**2])
        _ = util.simulateExpressionVector(vec, {}, is_plot=IS_PLOT)

    def testSimulateRR(self):
        util.simulateRR(is_plot=IS_PLOT)

    def testMakeSymbolDct(self):
        dct = util.makeSymbolDct(theta*k_d, PARAM_DCT, exclude_names=["t"])
        self.assertEqual(dct[theta], PARAM_DCT["theta"])

    def testFindSinusoidCoefficients(self):
        expression = sp.cos(t*theta)*k_d + sp.sin(t*theta)*k_d + 1
        dct = util.findSinusoidCoefficients(expression)
        for key in ["a", "b", "c"]:
            self.assertTrue(key in dct.keys())


if __name__ == "__main__":
    unittest.main()
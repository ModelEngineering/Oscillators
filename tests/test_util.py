'''Tests for util.py'''

from src.Oscillators import k_d, t, theta
from src.Oscillators.model import MODEL, PARAM_DCT 
from src.Oscillators import util
from src.Oscillators.util import TIMES, MODEL

import sympy as sp
import tellurium as te
import unittest

IGNORE_TEST = True
IS_PLOT = False

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
        if IGNORE_TEST:
            return
        vec = sp.Matrix([t, t**2])
        _ = util.simulateExpressionVector(vec, {}, is_plot=IS_PLOT)

    def testSimulateRR(self):
        if IGNORE_TEST:
            return
        util.simulateRR(is_plot=IS_PLOT)

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


if __name__ == "__main__":
    unittest.main()
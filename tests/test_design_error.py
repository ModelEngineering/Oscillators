from src.Oscillators.design_error import DesignError
from src.Oscillators.designer import Designer

import numpy as np
import unittest

IGNORE_TEST = False
IS_PLOT = False
REMOVE_FILES = []
END_TIME = 5

class TestDesignError(unittest.TestCase):

    def setUp(self):
        self.designer = Designer(theta=2*np.pi, alpha=3, phi=0, omega=5)
        self.design_error = DesignError(self.designer)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.design_error.designer.k2, self.designer.k2)

    def testCalculate(self):
        if IGNORE_TEST:
            return
        self.design_error.calculate()
        self.assertTrue(np.isclose(self.design_error.feasibledev, 0))
    
    def testEvaluatePredictions(self):
        if IGNORE_TEST:
            return
        fractional_error = self.design_error._evaluatePredictions()
        self.assertLess(fractional_error, 1e-5)

    def testLt(self):
        if IGNORE_TEST:
            return
        design_error = DesignError(self.designer)
        self.assertFalse(self.design_error < self.design_error)
        design_error.alphadev = 0
        design_error.phidev = 5
        design_error.feasibledev = 1
        self.design_error.feasibledev = 0
        self.design_error.alphadev = 5
        self.design_error.phidev = 0
        self.assertTrue(self.design_error < design_error)
        self.assertFalse(design_error < self.design_error)
        # alphadev
        design_error.feasibledev = 0
        self.assertFalse(self.design_error < design_error)
        # phidev
        self.design_error.alphadev = 0
        self.assertTrue(self.design_error < design_error)


if __name__ == "__main__":
    unittest.main()
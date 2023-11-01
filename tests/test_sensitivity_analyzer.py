import src.Oscillators.constants as cn
from src.Oscillators.sensitivity_analyzer import SensitivityAnalyzer

import numpy as np
import pandas as pd
import sympy as sp
import unittest

IGNORE_TEST = False
IS_PLOT = False

class TestSensitivityAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = SensitivityAnalyzer()

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.analyzer.symbol_dct, dict))
        self.assertTrue(isinstance(self.analyzer.baseline_df, pd.DataFrame))
        self.assertTrue(isinstance(self.analyzer.baseline_df.loc[cn.C_ALPHA, cn.C_X1], float))

    def testGetRandomValues(self):
        if IGNORE_TEST:
            return
        num_sample = 2
        values = self.analyzer._getRandomValues(cn.C_X1, cn.C_OMEGA, cv=1, num_sample=num_sample)
        self.assertEqual(len(values), num_sample)
        self.assertTrue(isinstance(values[0], float))
        
    def testInitializeTwoLevelDct(self):
        if IGNORE_TEST:
            return
        dct = self.analyzer._initializeTwoLevelDct()
        self.assertTrue(isinstance(dct, dict))
        self.assertTrue(isinstance(dct[cn.C_X1], dict))
        self.assertTrue(isinstance(dct[cn.C_X1][cn.C_THETA], list))

    def testMakeDataFrameFromTwoLevelDct(self):
        if IGNORE_TEST:
            return
        dct = self.analyzer._initializeTwoLevelDct()
        df = self.analyzer._makeDataFrameFromTwoLevelDct(dct)
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertEqual(df.shape, (4, 2))  


if __name__ == "__main__":
    unittest.main()
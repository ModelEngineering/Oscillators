import src.Oscillators.constants as cn
from src.Oscillators.sensitivity_analyzer import SensitivityAnalyzer
from src.Oscillators import sensitivity_analyzer as sa

import numpy as np
import os
import pandas as pda
import shutil
import sympy as sp
import unittest

IGNORE_TEST = True
IS_PLOT = False
ANALYZER = SensitivityAnalyzer()  # Used for debugging individual tests
TEST_DIR = os.path.dirname(os.path.abspath(__file__)) # This directory

class TestSensitivityAnalyzer(unittest.TestCase):

    def setUp(self):
        self.remove()
        if IGNORE_TEST:
            self.analyzer = ANALYZER
        else:
            self.analyzer = SensitivityAnalyzer()

    def tearDown(self):
        self.remove()

    def remove(self):
        temp_dir = sa.SENSITIVITY_DATA_DIR % TEST_DIR
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.analyzer.baseline_oc_value_df, pd.DataFrame))
        self.assertTrue(isinstance(self.analyzer.baseline_oc_value_df.loc[cn.C_ALPHA, cn.C_X1], float))

    def testGetRandomValues(self):
        if IGNORE_TEST:
            return
        num_sample = 20
        values = self.analyzer._getRandomValues(cn.C_K1, 0.1, num_sample=num_sample)
        self.assertEqual(len(values), num_sample)
        self.assertTrue(isinstance(values[0], float))
        new_values = self.analyzer._getRandomValues(cn.C_K1, 1, num_sample=num_sample)
        self.assertGreater(np.min(values), np.min(new_values))
        
    def testInitializeTwoLevelDct(self):
        if IGNORE_TEST:
            return
        dct = self.analyzer._initializeTwoLevelDct()
        self.assertTrue(isinstance(dct, dict))
        self.assertTrue(isinstance(dct[cn.C_X1], dict))
        self.assertTrue(isinstance(dct[cn.C_X1][cn.C_THETA], list))
        
    def testMakeRandomParameterDct(self):
        if IGNORE_TEST:
            return
        dct = self.analyzer._makeRandomParameterDct()
        self.assertTrue(isinstance(dct, dict))
        self.assertTrue(isinstance(dct[cn.C_X1], dict))
        self.assertTrue(isinstance(dct[cn.C_X1][cn.C_THETA], np.ndarray))

    def testMakeDataFrameFromTwoLevelDct(self):
        if IGNORE_TEST:
            return
        dct = self.analyzer._initializeTwoLevelDct()
        df = self.analyzer._makeDataFrameFromTwoLevelDct(dct)
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertEqual(df.shape, (4, 2))

    def testMakeErrorStatistics(self):
        if IGNORE_TEST:
            return
        statistics = self.analyzer.makeErrorStatistics(frac_deviation=0.5, num_sample=10)
        for attr in dir(statistics):
            if attr.startswith("_"):
                continue
            if "function" in attr:
                continue
            self.assertIsNotNone(getattr(statistics, attr))

    def testMakeData(self):
        #if IGNORE_TEST:
        #    return
        self.analyzer.makeData(frac_deviations=[0.1, 0.5], num_sample=10, data_dir=TEST_DIR)
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    unittest.main()


import src.Oscillators.constants as cn
from src.Oscillators.sensitivity_analyzer import SensitivityAnalyzer
from src.Oscillators import sensitivity_analyzer as sa

import numpy as np
import os
import pandas as pd
import shutil
import unittest

IGNORE_TEST = False
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
        if IGNORE_TEST:
            # Keep data if debugging
            return
        temp_dir = sa.SENSITIVITY_DATA_DIR % TEST_DIR
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.analyzer.baseline_oc_value_df, pd.DataFrame))
        self.assertTrue(self.analyzer.oc_expression_df.loc[cn.C_ALPHA, cn.C_X1] is not None)

    def testGetRandomValues(self):
        if IGNORE_TEST:
            return
        num_sample = 20
        values = self.analyzer._getRandomValues(cn.C_K1, 0.1, num_sample=num_sample)
        self.assertEqual(len(values), num_sample)
        trues = values >= 0
        self.assertTrue(np.all(trues))
        self.assertTrue(isinstance(values[0], float))
        
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
        self.assertTrue(isinstance(dct, dict))
        self.assertTrue(isinstance(dct[cn.C_K1], np.ndarray))

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
        statistics = self.analyzer.makeErrorStatistics(nrml_std=0.5, num_sample=10)
        for attr in dir(statistics):
            if attr.startswith("_"):
                continue
            if "function" in attr:
                continue
            self.assertIsNotNone(getattr(statistics, attr))

    def testMakeData(self):
        if IGNORE_TEST:
            return
        frac_deviations = [0.1, 0.5]
        statistic_types = ["mean", "std", "other"]
        self.analyzer.makeData(nrml_stds=[0.1, 0.5], num_sample=10, data_dir=TEST_DIR)
        for frac in frac_deviations:
            for stype in statistic_types:
                file_path = self.analyzer._getDataPath(stype, frac, data_dir=TEST_DIR)
                self.assertTrue(os.path.isfile(file_path))

    def testGetMetric(self):
        if IGNORE_TEST:
            return
        metric_dct = self.analyzer.getMetrics()
        trues = [k in cn.METRICS for k in metric_dct.keys()]
        self.assertTrue(np.all(trues))
        for k, df in metric_dct.items():
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertTrue(set(df.columns) == set([cn.C_MEAN, cn.C_STD]))


if __name__ == "__main__":
    unittest.main()
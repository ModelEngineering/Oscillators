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
        self.assertTrue(isinstance(self.analyzer.baseline_df.loc[cn.C_X1, cn.C_ALPHA], float))


if __name__ == "__main__":
    unittest.main()
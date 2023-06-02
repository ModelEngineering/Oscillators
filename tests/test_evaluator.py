import src.Oscillators.constants as cn
from src.Oscillators.evaluator import Evaluator
from src.Oscillators.designer import Designer

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import unittest

IGNORE_TEST = False
IS_PLOT = False
REMOVE_FILES = []
END_TIME = 5
EVALUATION_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_evaluation_data.csv")
TEST_EVALUATION_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_evaluation_data.csv")
REMOVE_FILES = [EVALUATION_CSV]

class TestEvaluator(unittest.TestCase):

    def init(self):
        self.remove()
        self.designer = Designer(theta=2*np.pi, alpha=3, phi=0, omega=5)
        self.evaluator = Evaluator(self.designer)

    def tearDown(self):
        self.remove()

    def remove(self):
        for path in REMOVE_FILES:
            if os.path.exists(path):
                os.remove(path)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.init()
        self.assertEqual(self.evaluator.designer.k2, self.designer.k2)
    
    def testMakeData(self):
        #if IGNORE_TEST:
        #    return
        self.init()
        df = self.evaluator.makeData(
              thetas=[0.1, 1.0],
              alphas=[0.1, 10.0], phis=[0, np.pi],
                            csv_path=EVALUATION_CSV)
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertTrue("alphadev" in df.columns)
        self.assertTrue("theta" in df.columns)
        self.assertTrue("alpha" in df.columns)
        self.assertTrue("phi" in df.columns)
        self.assertGreater(len(df), 0)
        self.assertTrue(os.path.exists(EVALUATION_CSV))

    def testPlotEvaluationData(self):
        if IGNORE_TEST:
            return
        self.init()
        self.evaluator.plotDesignErrors("feasibledev", plot_path="testPlotEvaluationData_feas.pdf", is_plot=IS_PLOT, vmin=-1, vmax=1,
                                         csv_path=TEST_EVALUATION_CSV)
        self.evaluator.plotDesignErrors("alphadev", plot_path="testPlotEvaluationData_alpha.pdf", vmin=-1, vmax=1, is_plot=IS_PLOT,
                                         csv_path=TEST_EVALUATION_CSV)
        self.evaluator.plotDesignErrors("phidev", plot_path="testPlotEvaluationData_phi.pdf", is_plot=IS_PLOT, vmin=-1, vmax=1,
                                         csv_path=TEST_EVALUATION_CSV)
        self.evaluator.plotDesignErrors(cn.C_PREDICTION_ERROR, plot_path="testPlotEvaluationData_prediction_error.pdf", is_plot=IS_PLOT, vmin=-1, vmax=1,
                                         csv_path=TEST_EVALUATION_CSV)

    def testPlotParameterHistograms(self):
        if IGNORE_TEST:
            return
        self.init()
        self.evaluator.plotParameterHistograms(csv_path=TEST_EVALUATION_CSV, output_path="testPlotParameterHistograms.pdf", is_plot=IS_PLOT)

if __name__ == "__main__":
    unittest.main()
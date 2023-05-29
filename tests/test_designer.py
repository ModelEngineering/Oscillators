from src.Oscillators.designer import Designer
from src.Oscillators import util

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import unittest

IGNORE_TEST = False
IS_PLOT = True
END_TIME = 5
EVALUATION_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_designer.csv")
# FIXME: Use TEST_EVALUATION_DATA
TEST_EVALUATION_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_data.csv")
REMOVE_FILES = [EVALUATION_CSV]

class TestOscillatorDesigner(unittest.TestCase):

    def setUp(self):
        self.remove()
        one_hertz = 2*np.pi
        self.designer = Designer(theta=one_hertz, alpha=3, phi=0, omega=5, end_time=END_TIME)

    def tearDown(self):
        self.remove()

    def remove(self):
        for path in REMOVE_FILES:
            if os.path.exists(path):
                os.remove(path)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        if IS_PLOT:
            df = pd.DataFrame({"time": self.designer.times, "xfit": self.designer.xfit_ref})
            df = df.set_index("time")
            util.plotDF(df, is_plot=False, output_path="testConstructor.pdf")
        self.assertEqual(len(self.designer.times), len(self.designer.xfit_ref))

    def testCalculateResiduals(self):
        if IGNORE_TEST:
            return
        parameters = lmfit.Parameters()
        parameters.add("k2", value=1.0, min=0.1)
        parameters.add("k4", value=1.0, min=0.1)
        parameters.add("k6", value=1.0, min=0.1)
        parameters.add("x1_0", value=1.0, min=0.1)
        parameters.add("x2_0", value=1.0, min=0.1)
        residuals = self.designer.calculateResiduals(parameters)
        if IS_PLOT:
            df = pd.DataFrame({"time": self.designer.times, "xfit": residuals[0:100], "xothers": residuals[100:]})
            df = df.set_index("time")
            util.plotDF(df, is_plot=False, output_path="testCalculateResiduals.pdf")    
        self.assertEqual(len(residuals), 2*self.designer.num_point)

    def testFind(self):
        if IGNORE_TEST:
            return
        _ = self.designer.find()
        if IS_PLOT:
            df = self.designer.simulate(is_plot=False)
            df["ref_xfit"] = self.designer.xfit_ref
            xfit_ref = self.designer.alpha*np.sin(self.designer.times*self.designer.theta + self.designer.phi) + self.designer.omega
            length = len(xfit_ref)
            UPPER = 0.25
            df["ref_xfit_calc"] = xfit_ref + np.random.normal(0, UPPER, length) 
            df["calc_xfit"] = self.designer.xfit + np.random.normal(0, UPPER, length)
            df["S1"] = df["S1"] + np.random.normal(0, UPPER, length)
            util.plotDF(df, is_plot=IS_PLOT, output_path="testFind.pdf")
        self.assertTrue(self.designer.minimizer.success)

    def testParameterToStr(self):
        if IGNORE_TEST:
            return
        stg = self.designer._parameterToStr("theta")
        self.assertTrue(isinstance(stg, str))


    def testPlotFit(self):
        if IGNORE_TEST:
            return
        designer = Designer(theta=2*np.pi, alpha=20, phi=-1,
                                omega=20, end_time=5)
        designer.find()
        if IS_PLOT:
            designer.plotFit(output_path="testPlotFit.pdf")

    def testPlotManyFits(self):
        if IGNORE_TEST:
            return
        self.designer.plotManyFits(output_path="testPlotManyFigs.pdf")

    def testEvaluate(self):
        if IGNORE_TEST:
            return
        evaluation = self.designer.evaluate()
        self.assertEqual(evaluation.feasibledev, 0)
        self.assertTrue(isinstance(evaluation.alphadev, float))

    def testMakeEvaluationData(self):
        if IGNORE_TEST:
            return
        df = self.designer.makeEvaluationData(
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
        self.designer.plotEvaluationData("feasibledev", plot_path="testPlotEvaluationData_feas.pdf", is_plot=IS_PLOT, vmin=-1, vmax=1,
                                         csv_path=TEST_EVALUATION_CSV)
        self.designer.plotEvaluationData("alphadev", plot_path="testPlotEvaluationData_alpha.pdf", vmin=-1, vmax=1, is_plot=IS_PLOT,
                                         csv_path=TEST_EVALUATION_CSV)
        self.designer.plotEvaluationData("phidev", plot_path="testPlotEvaluationData_phi.pdf", is_plot=IS_PLOT, vmin=-1, vmax=1,
                                         csv_path=TEST_EVALUATION_CSV)

    def testPlotParameterHistograms(self):
        if IGNORE_TEST:
            return
        self.designer.plotParameterHistograms(output_path="testPlotParameterHistograms.pdf", is_plot=IS_PLOT)


if __name__ == "__main__":
    unittest.main()
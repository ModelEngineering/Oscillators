from src.Oscillators.designer import Designer
from src.Oscillators import util
import src.Oscillators.constants as cn

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import unittest

IGNORE_TEST = True
IS_PLOT = False
END_TIME = 5


class TestOscillatorDesigner(unittest.TestCase):

    def setUp(self):
        self.designer = Designer(theta=2*np.pi, alpha=3, phi=0, omega=5)

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
        parameters.add(cn.C_K2, value=1.0, min=0.1)
        parameters.add(cn.C_K4, value=1.0, min=0.1)
        parameters.add(cn.C_K6, value=1.0, min=0.1)
        parameters.add(cn.C_X1_0, value=1.0, min=0.1)
        parameters.add(cn.C_X2_0, value=1.0, min=0.1)
        residuals = self.designer.calculateResiduals(parameters)
        if IS_PLOT:
            df = pd.DataFrame({"time": self.designer.times, "xfit": residuals[0:100], "xothers": residuals[100:]})
            df = df.set_index("time")
            util.plotDF(df, is_plot=False, output_path="testCalculateResiduals.pdf")    
        self.assertEqual(len(residuals), 2*self.designer.num_point)

    def testFind(self):
        if IGNORE_TEST:
            return
        _ = self.designer.parameterizeOscillator()
        if IS_PLOT:
            df = self.designer.simulate(is_plot=False)
            df["ref_xfit"] = self.designer.xfit_ref
            xfit_ref = self.designer.alpha*np.sin(self.designer.times*self.designer.theta + self.designer.phi) + self.designer.omega
            length = len(xfit_ref)
            UPPER = 0.25
            df["ref_xfit_calc"] = xfit_ref + np.random.normal(0, UPPER, length) 
            df["calc_xfit"] = self.designer.xfit + np.random.normal(0, UPPER, length)
            df[cn.C_S1] = df[cn.C_S1] + np.random.normal(0, UPPER, length)
            util.plotDF(df, is_plot=IS_PLOT, output_path="testFind.pdf")
        self.assertTrue(self.designer.minimizer.success)
        self.assertEqual(self.designer.design_error.feasibledev, 0)

    def testFindMiaximalOmega(self):
        #if IGNORE_TEST:
        #    return
        dct = self.designer.findMaximalOmega()
        self.designer.plotFit(start_time=0, end_time=100, num_point=1000)
        plt.savefig("testFindMaximalOmega.pdf")

    def testParameterToStr(self):
        if IGNORE_TEST:
            return
        stg = self.designer._parameterToStr("theta")
        self.assertTrue(isinstance(stg, str))

    def testPlotFit(self):
        if IGNORE_TEST:
            return
        designer = Designer(theta=2*np.pi, alpha=20, phi=-1,
                                omega=20)
        designer.parameterizeOscillator()
        designer.plotFit(output_path="testPlotFit.pdf", is_plot=IS_PLOT)

    def testPlotManyFits(self):
        if IGNORE_TEST:
            return
        self.designer.plotManyFits(output_path="testPlotManyFigs.pdf",
                                   is_plot=IS_PLOT)

    def testDesign(self):
        if IGNORE_TEST:
            return
        def make(**kwargs):
            designer = Designer.design(**kwargs)
            designer.parameterizeOscillator()
            return designer.design_error
        #
        design_error_x1 = make(is_x1=True)
        design_error_x2 = make(is_x1=False)
        design_error_both = make(is_both=True)
        for _ in range(3):
            new_design_error_both = make(is_both=True)
            if (new_design_error_both < design_error_both):
                design_error_both = new_design_error_both
        is_true = (design_error_both < design_error_x1) or (design_error_both < design_error_x2)
        self.assertTrue(is_true)


if __name__ == "__main__":
    unittest.main()
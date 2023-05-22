from src.Oscillators.oscillator_designer import OscillatorDesigner
from src.Oscillators import util

import lmfit
import matplotlib.pyplot as plt
import os
import pandas as pd
import unittest

IGNORE_TEST = True
IS_PLOT = True

class TestOscillatorDesigner(unittest.TestCase):

    def setUp(self):
        self.designer = OscillatorDesigner(theta=5, alphas=[1, 1], phis=[0, 1], omegas=[1, 1])

    def testConstructor(self):
        if IGNORE_TEST:
            return
        if IS_PLOT:
            df = pd.DataFrame({"time": self.designer.times, "x1": self.designer.x_refs[0], "x2": self.designer.x_refs[1]})
            df = df.set_index("time")
            util.plotDF(df, is_plot=False, output_path="testConstructor.pdf")
        self.assertEqual(len(self.designer.times), len(self.designer.x_refs[0]))

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
            df = pd.DataFrame({"time": self.designer.times, "x1": residuals[0:100], "x2": residuals[100:]})
            df = df.set_index("time")
            util.plotDF(df, is_plot=False, output_path="testCalculateResiduals.pdf")    
        self.assertEqual(len(residuals), 2*self.designer.num_point)

    def testFind(self):
        #if IGNORE_TEST:
        #    return
        result = self.designer.find()
        df = self.designer.simulate(is_plot=False, output_path="testFind.pdf")
        import pdb; pdb.set_trace() 


if __name__ == "__main__":
    unittest.main()
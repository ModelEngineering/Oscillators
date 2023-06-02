"""Calculates Errors in a Design"""

from src.Oscillators import t
from src.Oscillators import util
from src.Oscillators import solver

import numpy as np

SOLVER = solver.Solver()
SOLVER.solve()
MAX_FEASIBLEDEV = 1


class DesignError(object):

    def __init__(self, designer):
        self.designer = designer
        self.solver = SOLVER
        if designer.k2 is None:
            designer.find()
        # Outputs
        self.feasibledev = None
        self.alphadev = None
        self.phidev = None
        self.prediction_error = None

    def calculate(self):
        """Evaluates the fit.
        """
        # Check results of the finder
        if not self.designer.is_success:
            self.feasibledev = MAX_FEASIBLEDEV
            return
        # Completed the optimization
        oc1, oc2 = SOLVER.getOscillatorCharacteristics(dct=self.designer.params)
        if self.designer.is_x1:
            oc = oc1
        else:
            oc = oc2
        x_vec = util.getSubstitutedExpression(SOLVER.x_vec, self.designer.params) 
        x1_vec, x2_vec = x_vec[0], x_vec[1]
        x1_arr = np.array([float(x1_vec.subs({t: v})) for v in self.designer.times])
        x2_arr = np.array([x2_vec.subs({t: v}) for v in self.designer.times])
        arr = np.concatenate([x1_arr, x2_arr])
        self.feasibledev = sum(arr < -1e6)/len(arr)
        self.alphadev = oc.alpha/self.designer.alpha - 1
        self.phidev = self.designer.phi - oc.phi
        sign = np.sign(self.phidev)
        adj_phidev = min(np.abs(2*np.pi - np.abs(self.phidev)), np.abs(self.phidev))
        self.phidev = sign*adj_phidev/(2*np.pi)
        self.prediction_error = self._evaluatePredictions()
    
    def _evaluatePredictions(self):
        """
        Evaluates the predicted values of S1 and S2.

        Returns:
            float: fraction error
        """
        predicted_df = self.solver.simulate(param_dct=self.designer.params, expression=self.solver.x_vec, is_plot=False)
        simulated_df = util.simulateRR(param_dct=self.designer.params, end_time=self.designer.end_time,
                                     num_point=self.designer.num_point, is_plot=False)
        error_ssq = np.sum(np.sum(predicted_df - simulated_df)**2)
        total_ssq = np.sum(np.sum(simulated_df)**2)
        prediction_error = error_ssq/total_ssq
        return prediction_error

    def __lt__(self, other):
        """
        Checks if the design error of this object is less than another. Must have already called calculate().

        Args:
            other: DesignError
        Returns:
            bool
        """
        def isLessThan(x, y):
            if x is None:
                return False
            if y is None:
                return True
            return np.abs(x) < np.abs(y)
        #
        if isLessThan(self.feasibledev, other.feasibledev):
            return True
        if isLessThan(other.feasibledev, self.feasibledev):
            return False
        if isLessThan(self.alphadev, other.alphadev):
            return True
        if isLessThan(other.alphadev, self.alphadev):
            return False
        if isLessThan(self.phidev, other.phidev):
            return True
        if isLessThan(other.phidev, self.phidev):
            return False
        return self.designer.ssq < other.designer.ssq
    
    def __repr__(self):
        return "feasibledev=%s, alphadev=%s, phidev=%s, prediction_error=%s" % (
            self.feasibledev, self.alphadev, self.phidev, self.prediction_error)
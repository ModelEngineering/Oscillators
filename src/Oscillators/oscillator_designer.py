from src.Oscillators import util

import lmfit
import numpy as np
import pandas as pd


class OscillatorDesigner(object):

    LESS_THAN_ZERO_MULTIPLIER = 1e6

    def __init__(self, theta, alphas, phis, omegas, num_point=100):
        """
        Args:
            theta: float (frequency in radians)
            alphas: (float, float) (amplitude of the sinusoids)
            phis (_type_): _description_
            omegas (_type_): _description_
            num_point (int, optional): (number of points in a sinusoid series). Defaults to 100.
        """
        self.theta = theta
        self.alphas = alphas
        self.phis = phis
        self.omegas = omegas
        self.num_point = num_point
        #
        self.end_time = num_point/self.theta
        self.times = np.linspace(0, self.end_time, num_point)
        self.names = ["k2", "k4", "k6", "k_d", "x1_0", "x2_0"]
        # Reaction network parameters
        self.k1 = None
        self.k2 = None
        self.k3 = None
        self.k4 = None
        self.k5 = None
        self.k6 = None
        self.x1_0 = None
        self.x2_0 = None
        self.k_d = None
        # Reference sinusoids
        self.x_refs = np.array([alphas[n]*np.sin(self.times*self.theta + self.phis[n]) + self.omegas[n] for n in [0, 1]])

    def find(self):
        """
        Finds parameters of the reaction network that yield the desired Oscillator characeristics.
        """
        parameters = lmfit.Parameters()
        parameters.add("k2", value=1.0, min=0.1)
        parameters.add("k4", value=1.0, min=0.1)
        parameters.add("k6", value=1.0, min=0.1)
        parameters.add("x1_0", value=1.0, min=0.1)
        parameters.add("x2_0", value=1.0, min=0.1)
        #
        self.minimizer = lmfit.minimize(self.calculateResiduals, parameters, method="leastsq")
        # Set the results
        self.k1 = 1 # Aribratry choice
        self.k2 = self.minimizer.params["k2"].value
        self.k4 = self.minimizer.params["k4"].value
        self.k6 = self.minimizer.params["k6"].value
        self.x1_0 = self.minimizer.params["x1_0"].value
        self.x2_0 = self.minimizer.params["x2_0"].value
        self.k_d = self._calculateKd(self.k2)
        self.k3 = self.k1 + self.k2
        self.k5 = self.k3 + self.k_d
        return self.minimizer

    @staticmethod
    def _calculatePhaseOffset(theta):
        if np.abs(theta) > np.pi/2:
            phase_offset = np.pi
        else:
            phase_offset = 0
        return phase_offset
    
    def _calculateKd(self, k2):
        return self.theta**2/k2

    def calculateResiduals(self, params):
        """
        Calculates the results for the parameters.
        Parameters are: k2, k4 k6, x1_0, x2_0
        """
        k2 = params["k2"].value
        k4 = params["k4"].value
        k6 = params["k6"].value
        x1_0 = params["x1_0"].value
        x2_0 = params["x2_0"].value
        theta = self.theta
        k_d = self._calculateKd(k2)
        ####
        # x1
        ####
        numr_omega = -k2**2*k4 + k2**2*k6 - k2*k4*k_d + k6*theta**2
        denom = theta**2*(k2 + k_d)
        omega = numr_omega/denom
        #
        amp_1 = theta**2*(k2**2*x1_0 + k2**2*x2_0 - k2*k4 + k2*k_d*x1_0 - k4*k_d + theta**2*x2_0)**2 
        amp_2 = (k2**2*k4 - k2**2*k6 + k2*k4*k_d + k2*theta**2*x1_0 - k6*theta**2 + k_d*theta**2*x1_0)**2
        amp = np.sqrt(amp_1 + amp_2)/denom
        numr_phi = k2**2*k4 - k2**2*k6 + k2*k4*k_d + k2*theta**2*x1_0 - k6*theta**2 + k_d*theta**2*x1_0
        denom_phi = theta*(k2**2*x1_0 + k2**2*x2_0 - k2*k4 + k2*k_d*x1_0 - k4*k_d + theta**2*x2_0)
        phase_offset = self._calculatePhaseOffset(self.phis[0])
        phi = np.arctan(numr_phi/denom_phi) + phase_offset
        #
        x1 = amp*np.sin(self.times*theta + phi) + omega
        ####
        # x2
        ####
        denom = theta**2
        omega = (k2*k4 - k2*k6 + k4*k_d)/denom
        #
        amp_1 = theta**2*(k2*x1_0 + k2*x2_0 - k6 + k_d*x1_0)**2 + (k2*k4 - k2*k6 + k4*k_d - theta**2*x2_0)**2
        amp = np.sqrt(amp_1)/denom
        #
        phi = np.arctan((k2*k4 - k2*k6 + k4*k_d - theta**2*x2_0)/(theta*(k2*x1_0 + k2*x2_0 - k6 + k_d*x1_0)))
        phase_offset = self._calculatePhaseOffset(phi)
        phi = theta + phase_offset
        x2 = amp*np.sin(self.times*theta + phi) + omega
        # Calculate the residuals
        residual_arr = self.x_refs[0] - x1
        residual_arr = np.concatenate((residual_arr, self.x_refs[1] - x2))
        return residual_arr
    
    def simulate(self, **kwargs):
        """
        Does a roadrunner simulation of the reaction network found.

        Args:
            kwargs: dict (arguments to plotDF)
        Returns:
            pd.DataFrame
        """
        if self.k2 is None:
            raise ValueError("Must call find() before calling simulate()")
        #
        dct = {"k1": self.k1, "k2": self.k2, "k3": self.k3, "k4": self.k4, "k5": self.k5, "k6": self.k6,
                "S1": self.x1_0, "S2": self.x2_0}
        import pdb; pdb.set_trace()
        df = util.simulateRR(param_dct=dct, **kwargs)
        return df
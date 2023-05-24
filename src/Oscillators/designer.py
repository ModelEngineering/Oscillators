"""
Finds parameters of the network that give desired oscillating characteristics to S1 and
both S1 and S2 are feasible trajectories (non-negative concentrations).
"""

from src.Oscillators import util

import lmfit
import numpy as np
import pandas as pd


class Designer(object):

    LESS_THAN_ZERO_MULTIPLIER = 2

    def __init__(self, theta, alpha, phi, omega, end_time=5):
        """
        Args:
            theta: float (frequency in radians)
            alpha: float (amplitude of the sinusoids)
            phi: float (phase of the sinusoids)
            omega: float (offset of the sinusoids)
            num_point (int, optional): (number of points in a sinusoid series). Defaults to 100.
        """
        self.theta = theta
        self.alpha = alpha
        self.phi = phi
        self.omega = omega
        self.end_time = end_time
        self.num_point = 10*end_time
        #
        self.times = np.linspace(0, self.end_time, self.num_point)
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
        self.x1 = None
        # Reference sinusoids
        self.x1_ref = self.alpha*np.sin(self.times*self.theta + self.phi) + self.omega

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
        Calculates the results for the parameters. x1 residuals are calculated w.r.t. the reference.
        x2 residuals are calculated w.r.t. 0.

        Args:
            params: lmfit.Parameters 
                k2, k4 k6, x1_0, x2_0
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
        phase_offset = self._calculatePhaseOffset(self.phi)
        phi = np.arctan(numr_phi/denom_phi) + phase_offset
        #
        self.x1 = amp*np.sin(self.times*theta + phi) + omega
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
        phi = phi + phase_offset
        phi = phi + np.pi
        x2 = amp*np.sin(self.times*theta + phi) + omega
        x2_residuals = -1*(np.sign(x2)-1)*x2*self.LESS_THAN_ZERO_MULTIPLIER/2
        # Calculate the residuals
        residual_arr = self.x1_ref - self.x1
        residual_arr = np.concatenate([residual_arr, x2_residuals])
        # Updates the parameters
        self.k1 = 1 # Aribratry choice
        self.k2 = k2
        self.k_d = k_d
        self.k3 = self.k1 + k2
        self.k4 = k4
        self.k5 = self.k3 + k_d
        self.k6 = k6
        self.x1_0 = x1_0
        self.x2_0 = x2_0
        #
        return residual_arr

    @property 
    def params(self):
        return {"k1": self.k1, "k2": self.k2, "k3": self.k3, "k4": self.k4, "k5": self.k5, "k6": self.k6,
                "S1": self.x1_0, "S2": self.x2_0}

    
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
        df = util.simulateRR(param_dct=self.params, **kwargs)
        return df
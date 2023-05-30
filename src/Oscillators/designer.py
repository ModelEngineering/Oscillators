"""
Finds parameters of the network that give desired oscillating characteristics to S1 and
both S1 and S2 are feasible trajectories (non-negative concentrations).

BUGS
1. Poor fits for phi > pi/2
"""

from src.Oscillators import t
import src.Oscillators.constants as cn
from src.Oscillators import util
from src.Oscillators.solver import Solver

import collections
import matplotlib.pyplot as plt
import lmfit
import numpy as np
import os


INITIAL_SSQ = 1e8
MIN_VALUE = 0  # Minimum value for the parameters
MAX_VALUE = 1e3  # Maximum value for the parameters
MAX_RESIDUAL = 1e6
SOLVER = Solver()
SOLVER.solve()
EVALUATION_CSV = os.path.join(os.path.dirname(__file__), "evaluation_data.csv")
EVALUATION_PLOT_PATH = os.path.join(os.path.dirname(__file__), "evaluation_plot.pdf")
HISTOGRAM_PLOT_PATH = os.path.join(os.path.dirname(__file__), "histogram_plot.pdf")


class Evaluation(object):

    def __init__(self, feasibledev=None, alphadev=None, phidev=None, k2=None, k_d=None, k4=None,
                 k6=None, x1_0=None, x2_0=None):
        self.feasibledev = feasibledev
        self.alphadev = alphadev
        self.phidev = phidev
        self.k2 = k2
        self.k_d = k_d
        self.k4 = k4
        self.k6 = k6
        self.x1_0 = x1_0
        self.x2_0 = x2_0



class Designer(object):

    LESS_THAN_ZERO_MULTIPLIER = 2

    def __init__(self, theta, alpha, phi, omega, end_time=10, is_x1=True):
        """
        Args:
            theta: float (frequency in radians)
            alpha: float (amplitude of the sinusoids)
            phi: float (phase of the sinusoids)
            omega: float (offset of the sinusoids)
            num_point (int, optional): (number of points in a sinusoid series). Defaults to 100.
            is_x1: bool (True if the fit is for x1, False if the fit is for x2)
        """
        self.theta = theta
        self.alpha = alpha
        self.phi = phi
        self.omega = omega
        self.end_time = end_time
        self.num_point = 100
        self.is_x1 = is_x1
        #
        self.times = np.linspace(0, self.end_time, self.num_point)
        # Reaction network parameters
        self.k1 = None
        self.k2 = None
        self.k3 = None
        self.k4 = None
        self.k5 = None
        self.k6 = None
        self.k_d = None
        self.x1_0 = None
        self.x2_0 = None
        self.xfit = None # Fitted values. x1 if is_x1, x2 if not is_x1
        # Reference sinusoids
        self.ssq = INITIAL_SSQ  # Sum of squares calculated for the residuals
        self.xfit_ref = self.alpha*np.sin(self.times*self.theta + self.phi) + self.omega

    @property
    def _initial_value(self):
        return np.random.uniform(MIN_VALUE, MAX_VALUE)

    def find(self, num_tries=5):
        """
        Finds parameters of the reaction network that yield the desired Oscillator characeristics.
        """
        Result = collections.namedtuple("Result", ["params", "ssq", "minimizer"])
        #
        best_result = Result(params=dict(self.params), ssq=INITIAL_SSQ, minimizer=None)
        dct = {"k2": 0.1, "k_d": 0.1, "k4": 0.1, "k6": 0.1, "x1_0": 0.1, "x2_0": 0.1}
        self._setParameters(dct)
        #
        for _ in range(num_tries):
            self.ssq = INITIAL_SSQ
            parameters = lmfit.Parameters()
            parameters.add("k2", value=self._initial_value, min=MIN_VALUE, max=MAX_VALUE)
            parameters.add("k4", value=self._initial_value, min=MIN_VALUE, max=MAX_VALUE)
            parameters.add("k6", value=self._initial_value, min=MIN_VALUE, max=MAX_VALUE)
            parameters.add("x1_0", value=self._initial_value, min=MIN_VALUE, max=MAX_VALUE)
            parameters.add("x2_0", value=self._initial_value, min=MIN_VALUE, max=MAX_VALUE)
            minimizer = lmfit.minimize(self.calculateResiduals, parameters, method="leastsq")
            if minimizer.success:
                if self.ssq < best_result.ssq:
                    best_result = Result(params=dict(self.params), ssq=self.ssq, minimizer=minimizer)
        
        #
        self.minimizer = best_result.minimizer
        params = best_result.params
        dct = {n: params[n] for n in ["k2", "k4", "k6", "x1_0", "x2_0", "k_d"]}
        self._setParameters(dct)
        self.ssq = best_result.ssq
        #
        return self.minimizer

    @staticmethod
    def _calculatePhaseOffset(phi):
        if np.abs(phi) > np.pi:
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
        phi = phi + phase_offset
        phi = phi + np.pi
        x2 = amp*np.sin(self.times*theta + phi) + omega
        # Calculate residuals
        if self.is_x1:
            self.xfit = x1
            xother = x2
        else:
            self.xfit = x2
            xother = x1
        residual_arr = self.xfit_ref - self.xfit
        xother_residuals = -1*(np.sign(xother)-1)*x2*self.LESS_THAN_ZERO_MULTIPLIER/2
        residual_arr = np.concatenate([residual_arr, xother_residuals])
        # Updates the parameters
        ssq = np.sqrt(sum(residual_arr**2))
        if ssq < self.ssq:
            self.ssq = ssq
            dct = {"k2": k2, "k_d": k_d, "k4": k4, "k6": k6, "x1_0": x1_0, "x2_0": x2_0}
            self._setParameters(dct)
        #
        if np.isnan(residual_arr).any():
            residual_arr = np.nan_to_num(residual_arr, nan=MAX_RESIDUAL)
        return residual_arr
    
    def _setParameters(self, dct):
        """Sets values of the parameters

        Args:
            dct: dict
        """
        self.k1 = cn.K1_VALUE # Aribratry choice
        self.k2 = dct["k2"]
        self.k_d = dct["k_d"]
        if self.k2 is not None:
            self.k3 = self.k1 + self.k2
        self.k4 = dct["k4"]
        if self.k_d is not None:
            self.k5 = self.k3 + self.k_d
        self.k6 = dct["k6"]
        self.x1_0 = dct["x1_0"]
        self.x2_0 = dct["x2_0"]

    @property 
    def params(self):
        return {"k1": self.k1, "k2": self.k2, "k3": self.k3, "k4": self.k4, "k5": self.k5, "k6": self.k6,
                "S1": self.x1_0, "S2": self.x2_0, "x1_0": self.x1_0, "x2_0": self.x2_0, "k_d": self.k_d, "theta": self.theta}

    def simulate(self, **kwargs):
        """
        Does a roadrunner simulation of the reaction network found.

        Args:
            kwargs: dict (arguments to plotDF)
        Returns:
            pd.DataFrame
        """
        if self.k2 is None:
            self.find()
        #
        df = util.simulateRR(param_dct=self.params, num_point=self.num_point, end_time=self.end_time, **kwargs)
        return df
    
    def _parameterToStr(self, parameter_name, num_digits=4):
        expr = "r'$\%s$' + '=' + str(self.%s)[0:num_digits]" % (parameter_name, parameter_name)
        return eval(expr)
    
    def plotFit(self, ax=None, is_plot=True, output_path=None, xlabel="time", title=None,
                 is_legend=True, is_xaxis=True):
        """
        Plots the fit on the desired result.

        Args:
            kwargs: dict (arguments to plot options)
        """
        if self.k2 is None:
            self.find()
        #
        if ax is None:
            _, ax = plt.subplots()
        if title is None:
            titles = [self._parameterToStr(n) for n in ["theta", "alpha", "phi", "omega"]]
            title = ", ".join(titles)
        ax.set_title(title, fontsize=10)
        df = self.simulate()
        if self.is_x1:
            fit_label = "simulated S1"
            other_label = "simulated S2"
            predicted_label = "predicted S1"
            fit_values = df["S1"]
            other_values = df["S2"]
        else:
            fit_label = "simulated S2"
            other_label = "simulated S1"
            predicted_label = "predicted S2"
            fit_values = df["S2"]
            other_values = df["S1"]
        ax.plot(self.times, self.xfit_ref, label=fit_label, color="black")
        ax.scatter(self.times, fit_values, label="Fit", color="red")
        ax.plot(self.times, other_values, label=other_label, linestyle="--", color="grey")
        if is_xaxis:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])
        if is_legend:
            ax.legend([fit_label, predicted_label, other_label])
        if output_path is not None:
            ax.figure.savefig(output_path)
        if is_plot:
            plt.show()

    @classmethod
    def plotManyFits(cls, output_path=None, is_plot=True):
        """
        Constructs a grid of plots for the different parameters.

        Args:
            output_path: str
            is_plot: bool
        """
        alphas = [1, 1, 20, 20]
        length = len(alphas)
        thetas = 2*np.pi*np.array([1, 1, 2, 2])
        phis = np.pi*np.array([0, 0.2, 0, 0.2])
        omegas = [1, 2, 20, 30]
        nrow = length//2
        ncol = nrow
        fig, axs = plt.subplots(nrow, ncol) 
        irow = 0
        icol = 0
        for idx in range(length):
            designer = Designer(thetas[idx], alphas[idx], phis[idx], omegas[idx], end_time=2)
            designer.find()
            if irow == nrow - 1:
                is_xaxis = True
            else:
                is_xaxis = False
            designer.plotFit(is_plot=False, ax=axs[irow, icol], is_legend=False, is_xaxis=is_xaxis)
            icol += 1
            if icol == ncol:
                icol = 0
                irow += 1 
        if is_plot:
            plt.show()
        if output_path is not None:
            fig.savefig(output_path)
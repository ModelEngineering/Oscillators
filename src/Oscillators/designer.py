"""
Finds parameters of the network that give desired oscillating characteristics to S1 and
both S1 and S2 are feasible trajectories (non-negative concentrations).

TO DO:
1. Design for large omega but no other constraint.
"""

import src.Oscillators.constants as cn
from src.Oscillators.design_error import DesignError
from src.Oscillators import util
from src.Oscillators.solver import Solver

import collections
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import os


INITIAL_SSQ = 1e8
MIN_VALUE = 0  # Minimum value for the parameters
MAX_VALUE = 1e3  # Maximum value for the parameters
MAX_RESIDUAL = 1e6
MAX_FEASIBLEDEV = 1
SOLVER = Solver()
SOLVER.solve()
DEFAULT_DCT = {cn.C_X1_0: 1, cn.C_X2_0: 2}


class Designer(object):

    LESS_THAN_ZERO_MULTIPLIER = 2

    def __init__(self, theta=1, alpha=1, phi=0, omega=1, min_omega_other=0, is_x1=True):
        """
        Args:
            theta: float (frequency in radians)
            alpha: float (amplitude of the sinusoids)
            phi: float (phase of the sinusoids)
            omega: float (offset of the sinusoids)
            num_point (int, optional): (number of points in a sinusoid series). Defaults to 100.
            min_omega_other: float (minimum omega value for the other sinusoid)
            is_x1: bool (True if the fit is for x1, False if the fit is for x2)
        """
        self.theta = theta
        self.alpha = alpha
        self.phi = phi
        self.omega = omega
        self.min_omega_other = min_omega_other
        period = 2*np.pi/theta
        num_cycle = 10 
        self.end_time = num_cycle*period
        self.num_point = 1000
        self.is_x1 = is_x1
        self.solver = SOLVER
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
        #
        self.design_error = None

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
        self.design_error = DesignError(self)
        self.design_error.calculate()
        #
        return self.minimizer

    # FIXME: This neveer worked well 
    def findMaximalOmega(self, num_tries=5):
        """
        Finds parameters of the reaction network that yield the desired Oscillator characeristics.

        Args:
            num_tries: number of iterations
        Returns:
            lmfit.Parameters
        """
        Result = collections.namedtuple("Result", ["params", "ssq", "minimizer", "initialized"])
        #
        best_result = Result(params=dict(self.params), ssq=INITIAL_SSQ, minimizer=None, initialized=False)
        dct = {"k2": 0.1, "k_d": 0.1, "k4": 0.1, "k6": 0.1}
        self._setParameters(dct)
        #
        for _ in range(num_tries):
            self.ssq = INITIAL_SSQ
            parameters = lmfit.Parameters()
            parameters.add("k2", value=self._initial_value, min=MIN_VALUE, max=MAX_VALUE)
            parameters.add("k4", value=self._initial_value, min=MIN_VALUE, max=MAX_VALUE)
            parameters.add("k6", value=self._initial_value, min=MIN_VALUE, max=MAX_VALUE)
            minimizer = lmfit.minimize(self.calculateMaximalOmegaResiduals, parameters, method="leastsq")
            if minimizer.success:
                if (not best_result.initialized) or (self.ssq < best_result.ssq):
                    best_result = Result(params=dict(self.params), ssq=self.ssq, minimizer=minimizer, initialized=True)
        self._setParameters(best_result.params)
        for name, value in DEFAULT_DCT.items():
            if not name in self.params.keys():
                self.params[name] = value
        #
        return self.params

    def _calculatePhaseOffset(self, params, is_x1):
        """
        Calculates the phase offset for the sinusoid.
        
        Args:
            params: dict
            is_xi: bool (True if the fit is for x1, False if the fit is for x2)
        Returns:
            float
        """
        adjustment = 0
        k2, k4, k6, x1_0, x2_0, theta, k_d = self._getVariables(params)
        if is_x1:
            denom1 = k2*theta + k_d*theta
            denom2 = k2*theta**2 + k_d*theta**2
            numr1 = k2**2*x1_0 + k2**2*x2_0 - 2*k2*k4 + k2*k6 + k2*k_d*x1_0 - 2*k4*k_d
            numr2 = k2*k4*theta - k2*k6*theta + k4*k_d*theta
            total = numr1/denom1 + numr2/denom2 + theta*x2_0/(k2 + k_d)
            if total < 0:
                adjustment = np.pi
        else:
            total = (k2*x1_0 + k2*x2_0 - k6 + k_d*x1_0)/theta
            if total > 0:
                adjustment = np.pi
        return adjustment
    
    def _calculateKd(self, k2):
        return self.theta**2/k2
    
    def _getVariables(self, params):
        """
        Gets the variables from the parameters.
        Args:
            params: dict
        Returns:
            float*7 (k2, k4, k6, x1_0, x2_0, theta, k_d) 
        """
        def get(name):
            if name in params.keys():
                return params[name].value
            return DEFAULT_DCT[name]
        #
        k2 = params[cn.C_K2].value
        k4 = params[cn.C_K4].value
        k6 = params[cn.C_K6].value
        x1_0 = get(cn.C_X1_0)
        x2_0 = get(cn.C_X2_0)
        theta = self.theta
        k_d = self._calculateKd(k2)
        return k2, k4, k6, x1_0, x2_0, theta, k_d

    def calculateResiduals(self, params):
        """
        Calculates the results for the parameters. x1 residuals are calculated w.r.t. the reference.
        x2 residuals are calculated w.r.t. 0.

        Args:
            params: lmfit.Parameters 
                k2, k4 k6, x1_0, x2_0
        """
        k2, k4, k6, x1_0, x2_0, theta, k_d = self._getVariables(params)
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
        phase_offset = self._calculatePhaseOffset(params, is_x1=True)
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
        phase_offset = self._calculatePhaseOffset(params, is_x1=False)
        phi += phase_offset
        x2 = amp*np.sin(self.times*theta + phi) + omega
        # Calculate residuals
        if self.is_x1:
            self.xfit = x1
            xother = x2
        else:
            self.xfit = x2
            xother = x1
        name_dct = {cn.C_K2: k2, cn.C_K_D: k_d, cn.C_K4: k4, cn.C_K6: k6, cn.C_X1_0: x1_0, cn.C_X2_0: x2_0}
        # Calculate residuals
        residual_arr = self.xfit_ref - self.xfit
        xother_residuals = -1*(np.sign(xother-self.min_omega_other)-1)*xother*self.LESS_THAN_ZERO_MULTIPLIER/2
        #xother_residuals = -1*(np.sign(xother)-1)*xother*self.LESS_THAN_ZERO_MULTIPLIER/2
        residual_arr = np.concatenate([residual_arr, xother_residuals])
        # Updates the parameters
        ssq = np.sqrt(sum(residual_arr**2))
        if ssq < self.ssq:
            self.ssq = ssq
            self._setParameters(name_dct)
        #
        if np.isnan(residual_arr).any():
            residual_arr = np.nan_to_num(residual_arr, nan=MAX_RESIDUAL)
        return residual_arr

    def calculateMaximalOmegaResiduals(self, params):
        """
        Calculates the results for the parameters where the concern is maximizing the minimal omega.

        Args:
            params: lmfit.Parameters 
                k2, k4 k6, x1_0, x2_0
        """
        k2, k4, k6, _, _, theta, k_d = self._getVariables(params)
        def calcResidual(omega):
            sign = np.sign(omega)
            residual = sign*np.sign(max(omega, 1e-3))
            if residual < 0:
                omega *= self.LESS_THAN_ZERO_MULTIPLIER
            else:
                residual = 1/omega
            return residual
        ####
        # x1
        ####
        numr_omega = -k2**2*k4 + k2**2*k6 - k2*k4*k_d + k6*theta**2
        denom = theta**2*(k2 + k_d)
        omega1 = numr_omega/denom
        residual1 = calcResidual(omega1)
        ####
        # x2
        ####
        denom = theta**2
        omega2 = (k2*k4 - k2*k6 + k4*k_d)/denom
        residual2 = calcResidual(omega2)
        #
        residuals = np.repeat(residual1, 5)
        residuals = np.concatenate([residuals, np.repeat(residual2, 5)])
        return residuals
    
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
        if not "x1_0" in dct.keys():
            self.x1_0 = 1
        else:
            self.x1_0 = dct["x1_0"]
        if not "x2_0" in dct.keys():
            self.x2_0 = 2
        else:
            self.x2_0 = dct["x2_0"]

    @property 
    def params(self):
        return {"k1": self.k1, "k2": self.k2, "k3": self.k3, "k4": self.k4, "k5": self.k5, "k6": self.k6,
                "S1": self.x1_0, "S2": self.x2_0, "x1_0": self.x1_0, "x2_0": self.x2_0, "k_d": self.k_d, "theta": self.theta}

    def simulate(self, start_time=0, end_time=5, num_point=50):
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
        df = util.simulateRR(param_dct=self.params, start_time=start_time, num_point=num_point, end_time=end_time)
        return df
    
    def _parameterToStr(self, parameter_name, num_digits=4):
        expr = "r'$\%s$' + '=' + str(self.%s)[0:num_digits]" % (parameter_name, parameter_name)
        return eval(expr)
    
    def plotFit(self, ax=None, is_plot=True, output_path=None, xlabel="time", title=None,
                 is_legend=True, is_xaxis=True, **kwargs):
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
        df = self.simulate(**kwargs)
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
        else:
            plt.close()

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
            designer = Designer(theta=thetas[idx], alpha=alphas[idx],
                                 phi=phis[idx], omega=omegas[idx])
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
        else:
            plt.close()
        if output_path is not None:
            fig.savefig(output_path)

    def is_success(self):
        """Successful design."""
        return self.ssq != INITIAL_SSQ

    @classmethod
    def design(self, is_both=True, **kwargs):
        """Designs an oscillator with the desired characteristics.

        Args:
            is_both (bool): Consider designs with x1 and x2 as the oscillating species.
            kwargs: dict (arguments to Designer constructor)
        Returns:
            Designer
        """
        def makeDesign(**designKwargs):
            designer = Designer(**designKwargs)
            designer.find()
            return designer
        #
        if is_both:
            new_kwargs = dict(kwargs)
            if "is_x1" in kwargs:
                del new_kwargs["is_x1"]
            designer_x1 = makeDesign(is_x1=True, **new_kwargs) 
            designer_x2 = makeDesign(is_x1=False, **new_kwargs)
            if designer_x1.design_error < designer_x2.design_error:
                designer = designer_x1
            else:
                designer = designer_x2
        else:
            designer = Designer(**kwargs)
        return designer
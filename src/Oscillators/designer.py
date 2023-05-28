"""
Finds parameters of the network that give desired oscillating characteristics to S1 and
both S1 and S2 are feasible trajectories (non-negative concentrations).

BUGS
1. Poor fits for phi > pi/2
"""

from src.Oscillators import t
from src.Oscillators import util
from src.Oscillators.solver import Solver

import collections
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import lmfit
import numpy as np
import os
import pandas as pd
import seaborn as sns


INITIAL_SSQ = 1e8
MIN_VALUE = 0  # Minimum value for the parameters
MAX_VALUE = 1e5  # Maximum value for the parameters
MAX_RESIDUAL = 1e6
SOLVER = Solver()
SOLVER.solve()
CSV_PATH = os.path.join(os.path.dirname(__file__), "evaluation_data.csv")
PLOT_PATH = os.path.join(os.path.dirname(__file__), "evaluation_data.pdf")

Evaluation = collections.namedtuple("Evaluation", "feasibledev, alphadev, phidev, k2, k_d, k4, k6, x1_0, x2_0")


class Designer(object):

    LESS_THAN_ZERO_MULTIPLIER = 2

    def __init__(self, theta, alpha, phi, omega, end_time=10):
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
        self.num_point = 100
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
        self.x1 = None
        # Reference sinusoids
        self.ssq = INITIAL_SSQ  # Sum of squares calculated for the residuals
        self.x1_ref = self.alpha*np.sin(self.times*self.theta + self.phi) + self.omega

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
        self.k1 = 1 # Aribratry choice
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
        ax.plot(self.times, self.x1_ref, label="simulated S1", color="black")
        df = self.simulate()
        ax.scatter(self.times, df["S1"], label="Fit", color="red")
        ax.plot(self.times, df["S2"], label="smulated S2", linestyle="--", color="grey")
        if is_xaxis:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])
        if is_legend:
            ax.legend(["simulated S1", "Fitted S1", "simulated S2"])
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

    def evaluate(self):
        """Evaluates the fit.

        Returns:
            Evaluation
        """
        if self.k2 is None:
            self.find()
        # Check if success
        if self.ssq == INITIAL_SSQ:
            return Evaluation(feasibledev=False, alphadev=None, phidev=None)
        # Completed the optimization
        oc, _ = SOLVER.getOscillatorCharacteristics(dct=self.params)
        x_vec = util.getSubstitutedExpression(SOLVER.x_vec, self.params) 
        x1_vec, x2_vec = x_vec[0], x_vec[1]
        x1_arr = np.array([float(x1_vec.subs({t: v})) for v in self.times])
        x2_arr = np.array([x2_vec.subs({t: v}) for v in self.times])
        arr = np.concatenate([x1_arr, x2_arr])
        feasibledev = sum(arr < -1e6)/len(arr)
        alphadev = oc.alpha/self.alpha - 1
        phidev = self.phi - oc.phi
        sign = np.sign(phidev)
        phidev = sign*(sign*100*phidev % int(200*np.pi))/100.0
        phidev = phidev/2*np.pi
        return Evaluation(feasibledev=feasibledev, alphadev=alphadev, phidev=phidev,
                          k2=self.k2, k_d=self.k_d, k4=self.k4, k6=self.k6, x1_0=self.x1_0, x2_0=self.x2_0)
    
    @classmethod
    def plotEvaluationData(cls, value_name, csv_path=CSV_PATH, is_plot=True,
                           plot_path=PLOT_PATH, title=None, vmin=0, vmax=1):
        """Plots previously constructed evaluation data.
        Plots 4 heatmaps, one per phase. A heatmap as x = frequency, y=amplitude

        Args:
            value_name: str ("feasibledev", "ampdev", "phidev")
            csv_path: str
            output_path: str
            is_plot: bool
            plot_path: str (pdf file for plot)
            vmin: float (minimum on colobar)
            vmax: float (maximum on colobar)
        """
        df = pd.read_csv(csv_path)
        df = df.round(decimals=1)
        nrow = 2
        ncol = 2
        fig = plt.figure()
        grid_size = 8
        space = 2  # space between visual elements
        psize = 8  # size of a side of a plot
        bsize = 2  # size of a side of a colorbar
        width = 2*space + ncol*psize + bsize
        length = space + nrow*psize
        gs = GridSpec(length, width, figure=fig)
        plot_starts = [0, psize + space]
        plot_ends = [psize, 2*psize+ space]
        cbar_start = 2*(psize+space)
        cbar_end = cbar_start + 1
        #
        icol = 0
        irow = 0
        phis = df["phi"].unique()
        # Iterate across the plots
        for idx, phi in enumerate(phis):
            ax = fig.add_subplot(gs[plot_starts[irow]:plot_ends[irow], plot_starts[icol]:plot_ends[icol]])
            new_df = df[df["phi"] == phi]
            if (irow == nrow-1) and (icol == ncol - 1):
                cbar_ax = fig.add_subplot(gs[:, cbar_start:cbar_end])
                cbar=True
            else:
                cbar_ax = None
                cbar=False
            plot_df = pd.pivot_table(new_df, values=value_name, index='alpha', columns='theta')
            plot_df = plot_df.sort_index(ascending=False)
            g = sns.heatmap(plot_df, cmap="seismic", vmin=vmin, vmax=vmax, linewidths=1.0, annot=True, ax=ax, cbar=cbar, cbar_ax=cbar_ax,
                            annot_kws={"fontsize":6}, cbar_kws={'label': 'error fraction'})
            g.set_xticklabels(g.get_xticklabels(), rotation = 30, fontsize = 8)
            g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 8)
            ax.set_ylabel(r'$\alpha$')
            ax.set_title(r'$\phi$ = {}'.format(np.round(phi, 2)))
            icol += 1
            if irow == nrow - 1:
                is_xaxis = True
            else:
                is_xaxis = False
            if not is_xaxis:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                ax.set_xlabel(r'$\theta$')
            if icol == ncol:
                icol = 0
                irow += 1 
        #
        if plot_path is not None:
            fig.savefig(plot_path)
        if is_plot:
            plt.show()
    
    @classmethod
    def makeEvaluationData(cls, thetas=[0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0],
                            alphas=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0], phis=[0, 0.5*np.pi, np.pi, 1.5*np.pi],
                            csv_path=CSV_PATH, is_report=True):
        """
        Creates a CSV file with evaluation data using alpha=omega.

        Args:
            output_path: str
            is_plot: bool

        Returns:
            pd.DataFrame
                columns: theta, alpha, phi, feasibledev, alphadev, phidev, k2, k_d, k4, k6, x1_0, x2_0
        """
        num_rows = len(thetas)*len(alphas)*len(phis)
        #
        local_names = ["theta", "alpha", "phi"]
        evaluation_names = ["feasibledev", "alphadev", "phidev"]
        designer_names = ["k2", "k_d", "k4", "k6", "x1_0", "x2_0"]
        names = list(local_names)
        names.extend(evaluation_names)
        names.extend(designer_names)
        result_dct = {n: [] for n in names}
        count = 0
        percent = 0
        for theta in thetas:
            for alpha in alphas:
                for phi in phis:
                    designer = Designer(theta, alpha, phi, alpha)
                    evaluation = designer.evaluate()
                    for name in local_names:
                        stmt = "result_dct['%s'].append(%s)" % (name, name)
                        exec(stmt)
                    for name in evaluation_names:
                        stmt = "result_dct['%s'].append(evaluation.%s)" % (name, name)
                        exec(stmt)
                    for name in designer_names:
                        stmt = "result_dct['%s'].append(designer.%s)" % (name, name)
                        exec(stmt)
                    count += 1
                    if is_report:
                        new_percent = (100*count)//num_rows
                        if new_percent > percent:
                            percent = new_percent
                            msg = "Completed %d%%" % percent
                            print(msg)
        df = pd.DataFrame(result_dct)
        df.to_csv(csv_path, index=False)
        return df
    
    @staticmethod
    def addPathSuffix(path, suffix):
        parts = os.path.splitext(path)
        return parts[0] + suffix + parts[1]
    
if __name__ == "__main__":
    Designer.makeEvaluationData()
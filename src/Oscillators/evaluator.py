"""Evaluates a design."""

# Import packagesa
from src.Oscillators import t
from src.Oscillators import util
from src.Oscillators import constants as cn
from src.Oscillators.designer import Designer, MAX_VALUE, INITIAL_SSQ
from src.Oscillators import solver

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import numpy as np
import pandas as pd
import seaborn as sns


SOLVER = solver.Solver()
SOLVER.solve()
MAX_FEASIBLEDEV = 1
EVALUATION_CSV = os.path.join(os.path.dirname(__file__), "evaluation_data.csv")
EVALUATION_PLOT_PATH = os.path.join(os.path.dirname(__file__), "evaluation_plot.pdf")
HISTOGRAM_PLOT_PATH = os.path.join(os.path.dirname(__file__), "histogram_plot.pdf")
K1_VALUE = 1.0


class Evaluator(object):

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

    def evaluate(self):
        """Evaluates the fit.
        """
        # Check results of the finder
        if self.designer.ssq == INITIAL_SSQ:
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
    
    @classmethod
    def makeData(cls, thetas=[0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0],
                            alphas=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0], phis=[0, 0.5*np.pi, np.pi, 1.5*np.pi],
                            csv_path=EVALUATION_CSV, is_report=True, **kwargs):
        """
        Creates a CSV file with evaluation data using alpha=omega.

        Args:
            output_path: str
            is_plot: bool
            **kwargs: dict (arguments to Designer constructor)

        Returns:
            pd.DataFrame
                columns: theta, alpha, phi, feasibledev, alphadev, phidev, k2, k_d, k4, k6, x1_0, x2_0
        """
        num_rows = len(thetas)*len(alphas)*len(phis)
        #
        self_names = [cn.C_THETA, cn.C_ALPHA, cn.C_PHI]
        evaluator_names = [cn.C_FEASIBLEDEV, cn.C_ALPHADEV, cn.C_PHIDEV, cn.C_PREDICTION_ERROR]
        designer_names = [cn.C_K2, cn.C_K_D, cn.C_K4, cn.C_K6, cn.C_X1_0, cn.C_X2_0]
        names = list(designer_names)
        names.extend(evaluator_names)
        names.extend(self_names)
        result_dct = {n: [] for n in names}
        count = 0
        percent = 0
        for theta in thetas:
            for alpha in alphas:
                for phi in phis:
                    designer = Designer(theta, alpha, phi, alpha, **kwargs)
                    evaluator = Evaluator(designer)
                    evaluator.evaluate()
                    for name in self_names:
                        stmt = "result_dct['%s'].append(%s)" % (name, name)
                        exec(stmt)
                    for name in evaluator_names:
                        stmt = "result_dct['%s'].append(evaluator.%s)" % (name, name)
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
    
    @classmethod
    def plotEvaluationData(cls, value_name, csv_path=EVALUATION_CSV, is_plot=True,
                           plot_path=EVALUATION_PLOT_PATH, title=None, vmin=0, vmax=1):
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
                            annot_kws={"fontsize":6}, cbar_kws={'label': 'error fraction'}, linecolor="grey")
            g.set_xticklabels(g.get_xticklabels(), rotation = 30, fontsize = 8)
            g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 8)
            ax.set_ylabel(r'$\alpha$')
            ax.set_title(r'$\phi$ = {}'.format(np.round(phi, 2)))
            if icol > 0:
                ax.set_yticklabels([])
                ax.set_yticks([])
                ax.set_ylabel("")
            if irow < nrow - 1:
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_xlabel("")
            if irow == nrow - 1:
                is_xaxis = True
            else:
                is_xaxis = False
            if not is_xaxis:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                ax.set_xlabel(r'$\theta$')
            icol += 1
            if icol == ncol:
                icol = 0
                irow += 1 
        #
        if plot_path is not None:
            fig.savefig(plot_path)
        if is_plot:
            plt.show()

    @classmethod
    def plotParameterHistograms(cls, csv_path=EVALUATION_CSV, is_plot=True, output_path=HISTOGRAM_PLOT_PATH):
        """Plots histograms of kinetic parameters.

        Args:
            evaluation_data: str (path to evaluation data)
        """
        df = pd.read_csv(csv_path)
        df[cn.C_K1] = cn.K1_VALUE
        df[cn.C_K3] = df[cn.C_K2] + df[cn.C_K2]
        df[cn.C_K5] = df[cn.C_K3] + df[cn.C_K_D]
        nrow = 2
        ncol = 4
        fig = plt.figure()
        irow = 0
        icol = 0
        bins = np.linspace(0, MAX_VALUE, 20)
        gs = GridSpec(nrow, ncol, figure=fig)
        for name in cn.C_MODEL_PARAMETERS:
            ax = fig.add_subplot(gs[irow, icol])
            counts, _ = np.histogram(df[name], bins=bins)
            fractions = counts/sum(counts)
            xv = bins[0:-1]
            ax.bar(xv, fractions, width=bins[1]-bins[0])
            if name[0] == "x":
                display_name = name[0] + name[1] + " (0)"
            else:
                display_name = name
            display_name = display_name[0] + "_" + display_name[1:]
            display_name = display_name.replace("__", "")
            ax.set_title('$%s$' % display_name)
            ax.set_xlim([0, MAX_VALUE])
            ax.set_xticklabels(ax.get_xticklabels(), fontsize = 8)
            ax.set_ylim([0, 1])
            if irow < nrow - 1:
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_xlabel("")
            else:
                ax.set_xlabel("value")
            if icol > 0:
                ax.set_yticklabels([])
                ax.set_yticks([])
                ax.set_ylabel("")
            else:
                ax.set_ylabel("fraction")
            icol += 1
            if icol == ncol:
                icol = 0
                irow += 1
        if output_path is not None:
            fig.savefig(output_path)
        if is_plot:
            plt.show()
    
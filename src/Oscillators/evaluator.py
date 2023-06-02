"""Evaluates a design."""

# Import packagesa
from src.Oscillators import t
from src.Oscillators import util
from src.Oscillators import constants as cn
from src.Oscillators.designer import Designer, MAX_VALUE, INITIAL_SSQ
from src.Oscillators import solver

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import numpy as np
import pandas as pd
import seaborn as sns


SOLVER = solver.Solver()
SOLVER.solve()
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

    @classmethod
    def makeData(cls, thetas=[0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0],
                            alphas=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
                            #phis=[0, 0.5*np.pi, np.pi, 1.5*np.pi],
                            phis=[0, np.pi/2, np.pi, 3*np.pi/2],
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
        local_names = [cn.C_THETA, cn.C_ALPHA, cn.C_PHI]
        designer_names = [cn.C_K2, cn.C_K_D, cn.C_K4, cn.C_K6, cn.C_X1_0, cn.C_X2_0, cn.C_IS_X1]
        design_error_names = [cn.C_FEASIBLEDEV, cn.C_ALPHADEV, cn.C_PHIDEV, cn.C_PREDICTION_ERROR]
        names = list(designer_names)
        names.extend(design_error_names)
        names.extend(local_names)
        result_dct = {n: [] for n in names}
        count = 0
        percent = 0
        for theta in thetas:
            for alpha in alphas:
                for phi in phis:
                    designer = Designer(theta=theta, alpha=alpha, phi=phi, omega=alpha, **kwargs)
                    designer.find()
                    for name in local_names:
                        stmt = "result_dct['%s'].append(%s)" % (name, name)
                        exec(stmt)
                    for name in design_error_names:
                        stmt = "result_dct['%s'].append(designer.design_error.%s)" % (name, name)
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
    def plotDesignErrors(cls, value_name, csv_path=EVALUATION_CSV, is_plot=True,
                           plot_path=EVALUATION_PLOT_PATH, title=None, vmin=0, vmax=1):
        """Plots previously constructed evaluation data.
        Plots 4 heatmaps, one per phase. A heatmap as x = frequency, y=amplitude

        Args:
            value_name: str (in cn.DESIGN_ERROR_LABEL_DCT)
            csv_path: str
            output_path: str
            is_plot: bool
            plot_path: str (pdf file for plot)
            vmin: float (minimum on colobar)
            vmax: float (maximum on colobar)
        """
        if not value_name in cn.DESIGN_ERROR_LABEL_DCT.keys():
            raise ValueError("value_name %s not in %s" % (value_name, cn.DESIGN_ERROR_LABEL_DCT.keys()))
        #
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
        if len(phis) != nrow*ncol:
            raise RuntimeError("Expected %d phis, but found %d" % (nrow*ncol, len(phis)))
        # Iterate across the plots
        for phi in phis:
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
            cbar_label = cn.DESIGN_ERROR_LABEL_DCT[value_name]
            g = sns.heatmap(plot_df, cmap="seismic", vmin=vmin, vmax=vmax, linewidths=1.0, annot=True, ax=ax, cbar=cbar, cbar_ax=cbar_ax,
                            annot_kws={"fontsize":6}, cbar_kws={'label': cbar_label}, linecolor="grey")
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
        df[cn.C_S1] = df[cn.C_X1_0]
        df[cn.C_S2] = df[cn.C_X2_0]
        nrow = 2
        ncol = 4
        fig = plt.figure()
        irow = 0
        icol = 0
        bins = np.linspace(0, MAX_VALUE, 20)
        gs = GridSpec(nrow, ncol, figure=fig)
        for name in cn.C_SIMULATION_PARAMETERS:
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
            ticks_loc = ax.get_xticks().tolist()
            ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
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
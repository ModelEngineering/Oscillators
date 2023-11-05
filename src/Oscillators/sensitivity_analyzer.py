"""Analyzes the sensitivity of the 2SHO to variations in parameter values."""

import src.Oscillators.constants as cn
from src.Oscillators import util
from src.Oscillators.solver import Solver

import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sympy as sp

X_TERMS = [cn.C_X1, cn.C_X2]
PARAMETERS = [p for p in cn.ALL_PARAMETERS if not p in [cn.C_T, cn.C_X1, cn.C_X2]]
NUM_SAMPLE = 400

SENSITIVITY_DATA_DIR = os.path.join("%s", "sensitivity_data")
DEVIATION_DIR = os.path.join(SENSITIVITY_DATA_DIR, "%s")
MEAN_PATH = os.path.join(DEVIATION_DIR, "mean.csv")
STD_PATH = os.path.join(DEVIATION_DIR, "std.csv")
OTHER_PATH = os.path.join(DEVIATION_DIR, "other.csv") 
NRML_STDS = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
NRML_STDS.extend([0.11, 0.13, 0.15, 0.17, 0.19, 0.2])


ErrorStatistic = collections.namedtuple("ErrorStatistic",
        "mean_df std_df frac_nonoscillating, frac_infeasible sample_size")


class SensitivityAnalyzer(object):

    def __init__(self, baseline_parameter_dct=None):
        """Constructor.

        Args:
            parameter_dct: (dict) keys are parameter names, values are parameter values
        """
        if baseline_parameter_dct is None:
            baseline_parameter_dct = dict(cn.PARAM_DCT)
            baseline_parameter_dct[cn.C_X1_0] = baseline_parameter_dct[cn.C_S1]
            baseline_parameter_dct[cn.C_X2_0] = baseline_parameter_dct[cn.C_S2]
            del baseline_parameter_dct[cn.C_S1]
            del baseline_parameter_dct[cn.C_S2]
        self.baseline_parameter_dct = dict(baseline_parameter_dct)
        Solver.calculateDependentParameters(self.baseline_parameter_dct)
        self.solver = Solver()
        self.solver.solve()
        self.oc_expression_df, self.baseline_oc_value_df = self.makeOscillationCharacteristicsDF()

    def makeOscillationCharacteristicsDF(self):
        """
        Creates baseline OCs based on the parameter values.

        Returns:
            pd.DataFrame (symbolic solution)
                index: C_X1, C_X2
                columns: C_THETA, C_ALPHA, C_PHI, C_OMEGA
                values: expressions
            pd.DataFrame
                columns: C_X1, C_X2
                index: C_THETA, C_ALPHA, C_PHI, C_OMEGA
                values: float
        """
        oc1, oc2 = self.solver.getOscillatorCharacteristics()
        arr1 = np.array([oc1.theta, oc1.alpha, oc1.phi, oc1.omega])
        arr2 = np.array([oc2.theta, oc2.alpha, oc2.phi, oc2.omega])
        oc_df = pd.DataFrame([arr1, arr2], index=X_TERMS,
                          columns=[cn.C_THETA, cn.C_ALPHA, cn.C_PHI, cn.C_OMEGA])
        baseline_df = oc_df.copy()
        baseline_symbol_dct = util.makeSymbolDct(self.solver.x_vec, self.baseline_parameter_dct)
        # Calculate the baseline values for the oscillation characteristics
        for idx in oc_df.index:
            for column in oc_df.columns:
                value = float(sp.N(oc_df.loc[idx, column].subs(baseline_symbol_dct)))
                baseline_df.loc[idx, column] = value
        return oc_df.T, baseline_df.T
    
    def _depreacatedGetRandomValues(self, x_term, parameter_name, cv, num_sample):
        """Returns a random value of the parameter"""
        std = self.baseline_oc_value_df.loc[parameter_name, x_term]*cv
        result = np.random.normal(self.baseline_parameter_df.loc[parameter_name, x_term], std, num_sample)
        return result

    def _getRandomValues(self, parameter_name, nrml_std, num_sample):
        """Returns a random value of the parameter. Truncates at 0."""
        mean = self.baseline_parameter_dct[parameter_name]
        samples = np.random.normal(mean, mean*nrml_std, 10*num_sample)   #Over sample
        samples = samples[samples >= 0]
        if len(samples) < num_sample:
            raise ValueError("Insufficient samples")
        return samples[:num_sample]

    def _futureGetRandomValues(self, parameter_name, nrml_std, num_sample):
        """Returns a random value of the parameter"""
        lower = self.baseline_parameter_dct[parameter_name]*(1 - nrml_std)
        upper = self.baseline_parameter_dct[parameter_name]*(1 + nrml_std)
        result = np.random.uniform(lower, upper, num_sample)
        return result

    def _initializeTwoLevelDct(self):
        """Initializes a 2D dictionary"""
        dct = {}
        for x_term in X_TERMS:
            dct[x_term] = {n: [] for n in cn.OSCILLATION_CHARACTERISTICS}
        return dct

    def _makeDataFrameFromTwoLevelDct(self, dct):
        """
        Converts a 2D dictionary to a DataFrame

        Args:
            dct: dict (two level)
                one value in each position
        Returns:
            pd.DataFrame
                index: cn.OSCILLATION_CHARACTERISTICS
                columns: cn.C_X1, cn.C_X2
                values: float
        
        """
        new_dct = {oc: [] for oc in cn.OSCILLATION_CHARACTERISTICS}
        for x_term in X_TERMS:
            for oc in cn.OSCILLATION_CHARACTERISTICS:
                new_dct[oc].append(dct[x_term][oc])
        df = pd.DataFrame(new_dct, index=X_TERMS)
        return df.T
    
    def _makeRandomParameterDct(self, nrml_std=0.1, num_sample=1000):
        """
        Creates random sample of parameter values sampling from a mean adjusted distribution.

        Args:
            nrml_std: float (normalized standard deviation, coefficient of variation)
            num_sample: int (number of samples to generate)

        Returns:
            dict
                key: str (parameter name)
                value: np.ndarray (random sample of parameter values)
        """
        return {p: self._getRandomValues(p, nrml_std, num_sample=num_sample) for p in PARAMETERS}

    def makeErrorStatistics(self, nrml_std=0.1, num_sample=NUM_SAMPLE):
        """
        Calculates the average absolute error of each OC for the coefficient of variation.

        Args:
            nrml_std: float (normalized standard deviation, coefficient of variation)
        Returns:
            ErrorStatistic
                mean_df - mean absolute error for each OC
                std_df - std of absolute error for each OC (not std of the mean)
                num_negative - number of infeasible solutions
                num_nonoscillating - number of non-oscillating solutions
                sample_size - number of samples
        """
        parameter_sample_dct = self._makeRandomParameterDct(nrml_std=nrml_std, num_sample=num_sample)
        # Obtain samples of oscillation characteristics from the sampled parameter values
        oc_sample_dct = self._initializeTwoLevelDct()
        num_nonoscillating = 0
        for idx in range(num_sample):
            parameter_dct = {p: parameter_sample_dct[p][idx] for p in PARAMETERS}
            Solver.calculateK3K5KDTHETA(parameter_dct)
            symbol_dct = util.makeSymbolDct(self.solver.x_vec, parameter_dct)
            for x_term in X_TERMS:
                # Check for negative concentrations
                if parameter_dct[cn.C_K3] > parameter_dct[cn.C_K5]:
                    num_nonoscillating += 1
                    continue
                for oc in cn.OSCILLATION_CHARACTERISTICS:
                    # Calculate the oscillation characteristic
                    sample_value = float(sp.N(self.oc_expression_df.loc[oc, x_term].subs(symbol_dct)))
                    oc_sample_dct[x_term][oc].append(sample_value)
        # Calculate instances of negative concentrations, which are infeasible
        negative_arr = np.repeat(0, num_sample)
        for x_term in X_TERMS:
            amplitude_arr = np.array(oc_sample_dct[x_term][cn.C_ALPHA])
            omega_arr = np.array(oc_sample_dct[x_term][cn.C_OMEGA])
            negative_arr += amplitude_arr > omega_arr
        num_negative = np.sum(negative_arr > 0)
        # Calculate the average absolute error
        mean_dct = self._initializeTwoLevelDct()
        std_dct = self._initializeTwoLevelDct()
        for x_term in X_TERMS:
            for oc in cn.OSCILLATION_CHARACTERISTICS:
                baseline_value = self.baseline_oc_value_df.loc[oc, x_term]
                # Phase error is in units of radians
                if oc == cn.C_PHI:
                    divisor = 1
                else:
                    divisor = baseline_value
                abs_error_arr = np.abs((np.array(oc_sample_dct[x_term][oc]) - baseline_value)/divisor)
                mean_dct[x_term][oc] = np.mean(abs_error_arr)
                std_dct[x_term][oc] = np.std(abs_error_arr)
        # Construct the result
        mean_df = self._makeDataFrameFromTwoLevelDct(mean_dct)
        std_df = self._makeDataFrameFromTwoLevelDct(std_dct)
        frac_infeasible = (num_nonoscillating + num_negative)/num_sample
        return ErrorStatistic(mean_df=mean_df, std_df=std_df, frac_nonoscillating=num_nonoscillating/num_sample,
                              frac_infeasible=frac_infeasible, sample_size=num_sample)

    @staticmethod 
    def _getDataPath(statistic, frac_deviation, data_dir):
        """
        Returns the path to the data file.

        Args:
            statistic: str
            frac_deviation: float
            data_dir: str

        Returns:
            str
        """
        MEAN = "mean"
        STD = "std"
        OTHER = "other"
        PATH_DIR = {MEAN: MEAN_PATH % (data_dir, str(frac_deviation)),
                        STD: STD_PATH % (data_dir, str(frac_deviation)),
                        OTHER: OTHER_PATH % (data_dir, str(frac_deviation))
                        }
        if statistic not in PATH_DIR.keys():
            raise ValueError("Invalid path_type: %s" % statistic)
        return PATH_DIR[statistic]

    @classmethod
    def getSensitivityData(cls, statistic, nrml_std, data_dir):
        """
        Returns the path to the data file.

        Args:
            statistic: str (mean, std, other)
            nrml_std: float
            data_dir: str

        Returns:
            DataFrame or Series
        """
        path = cls._getDataPath(statistic, nrml_std, data_dir)
        return pd.read_csv(path, index_col=0)
    
    def makeData(self, nrml_stds, num_sample=NUM_SAMPLE, data_dir=cn.DATA_DIR, is_overwrite=False, is_report=True):
        """
        Creates the data needed for plotting.

        Args:
            nrml_stds: list-float (normal standard deviation -- coefficient of variation)
            num_sample: int (number of samples to generate)
            is_overwrite: bool (True to overwrite existing data)
            is_report: bool (True to report progress)

        Notes
            Retrive data with index using:  pd.read_csv(path_dir[MEAN], index_col=0)
        """
        MEAN = "mean"
        STD = "std"
        OTHER = "other"
        cur_dir = SENSITIVITY_DATA_DIR % data_dir
        if not os.path.isdir(cur_dir):
            os.mkdir(cur_dir)
        for nrml_std in nrml_stds:
            sub_dir = os.path.join(cur_dir, str(nrml_std))
            if not os.path.isdir(sub_dir):
                 os.mkdir(sub_dir)
            elif not is_overwrite:
                print("** Skipping %s" % str(nrml_std))
                continue
            print("** Processing %s" % str(nrml_std))
            statistics = self.makeErrorStatistics(nrml_std=nrml_std, num_sample=num_sample)
            statistics.mean_df.to_csv(self._getDataPath(MEAN, nrml_std, data_dir))
            statistics.std_df.to_csv(self._getDataPath(STD, nrml_std, data_dir))
            other_ser = pd.Series([statistics.frac_nonoscillating,
                                     statistics.frac_infeasible,
                                     statistics.sample_size], index=[cn.C_NONOSCILLATING,
                                                                     cn.C_INFEASIBLE,
                                                                     cn.C_SAMPLE_SIZE])
            other_ser.to_csv(self._getDataPath(OTHER, nrml_std, data_dir))

    @classmethod
    def getMetrics(cls):
        """
        Returns:
            dict: str (cn.METRICS)
            value: pd.DataFrame
                index: float (normalized standard deviation)
                columns: str (mean, std)
                values: float
        """
        result_dct = {n: {cn.C_MEAN: [], cn.C_STD: []} for n in cn.METRICS}
        others = [cn.C_NONOSCILLATING, cn.C_INFEASIBLE, cn.C_SAMPLE_SIZE]
        for nrml_std in NRML_STDS:
            mean_df = cls.getSensitivityData(cn.C_MEAN, nrml_std, cn.DATA_DIR)
            std_df = cls.getSensitivityData(cn.C_STD, nrml_std, cn.DATA_DIR)
            other_df = cls.getSensitivityData(cn.C_OTHER, nrml_std, cn.DATA_DIR)
            # Add the statistics
            for metric in cn.METRICS:
                if metric in others:
                    value = other_df.loc[metric, '0']
                    if metric in [cn.C_NONOSCILLATING, cn.C_INFEASIBLE]:
                        std = np.sqrt(value*(1-value))/np.sqrt(NUM_SAMPLE)
                    else:
                        std = 0
                    result_dct[metric][cn.C_MEAN].append(value)
                    result_dct[metric][cn.C_STD].append(std)
                else:
                    if metric != cn.C_THETA:
                        column = "x%s" % metric[-1]
                        idx = metric[:-1]
                    else:
                        column = "x1"
                        idx = metric
                    result_dct[metric][cn.C_MEAN].append(mean_df.loc[idx, column])
                    result_dct[metric][cn.C_STD].append(std_df.loc[idx, column]/np.sqrt(NUM_SAMPLE))
        # Convert to DataFrame
        for metric in cn.METRICS:
            result_dct[metric] = pd.DataFrame(result_dct[metric], index=NRML_STDS, columns=[cn.C_MEAN, cn.C_STD])
        #
        return result_dct
    
    def _plotMetric(self, metric, metric_df, ax=None, is_plot=True):
        """
        Plots a single metric.

        Args:
            metric: str (one of cn.METRICS)
            metric:df: pd.DataFrame
            ax: plt.Axes
        """
        AXIS_FONT_SIZE = 14
        TICKLABEL_FONT_SIZE = 14
        TITLE_FONT_SIZE = 18
        LABEL1 = "|relative error|"
        LABEL2 = "|radians|"
        LABEL3 = "probability"
        title_dct = {cn.C_ALPHA1: r"$\alpha_1$", cn.C_PHI1: r"$\phi_1$", cn.C_OMEGA1: r"$\omega_1$",
                    cn.C_THETA: r"$\theta$", cn.C_ALPHA2: r"$\alpha_2$", cn.C_PHI2: r"$\phi_2$", cn.C_OMEGA2: r"$\omega_2$"}
        ylabel_dct = {cn.C_ALPHA1: LABEL1, cn.C_PHI1: LABEL2, cn.C_OMEGA1: LABEL1,
                    cn.C_ALPHA2: LABEL1, cn.C_PHI2: LABEL2, cn.C_OMEGA2: LABEL1,
                    cn.C_THETA: LABEL1, cn.C_INFEASIBLE: LABEL3, cn.C_NONOSCILLATING: LABEL3}
        if ax is None:
            _, ax = plt.subplots(1)
        plot_df = metric_df.copy()
        if metric == cn.C_INFEASIBLE:
            plot_df[cn.C_MEAN] = 1 - metric_df[cn.C_MEAN]
        plot_df[cn.C_STD] = 2*plot_df[cn.C_STD]
        plot_df.plot(ax=ax, y=cn.C_MEAN, yerr=cn.C_STD, marker="o")
        x_pos = NRML_STDS[4]
        y_pos = 0.9
        if metric in title_dct.keys():
            text = title_dct[metric]
        else:
            if metric == cn.C_INFEASIBLE:
              text = "feasibility"
            else:
              text = metric
        ax.text(x_pos, y_pos, text, fontsize=TITLE_FONT_SIZE)
        ax.text(NRML_STDS[0], 0.6, ylabel_dct[metric], fontsize=AXIS_FONT_SIZE, rotation=90)
        xlabels = ax.get_xticklabels()
        ax.set_xticklabels(xlabels, fontsize=TICKLABEL_FONT_SIZE, rotation=-30)
        ax.set_xlabel("nrml std", fontsize=AXIS_FONT_SIZE)
        ax.set_ylim([0, 1])
        ylabels = ax.get_yticklabels()
        ax.set_yticklabels(ylabels, fontsize=TICKLABEL_FONT_SIZE)
        ax.legend([])
        if is_plot:
            plt.show()

    # FIXME: (a) y axis label inside of yaxis; (b) greek letters for titles
    def plotMetrics(self, is_plot=True):
        """
        Plots the metrics.

        Args:
            is_plot: bool
        """
        metric_dct = self.getMetrics()
        nrow = 2
        ncol = 4
        irow = 0
        icol = 0
        _, axes = plt.subplots(nrow, ncol, figsize=(15, 15))
        metrics = [cn.C_INFEASIBLE, 'alpha1', 'phi1', 'omega1', 'theta', 'alpha2', 'phi2', 'omega2']
        for metric in metrics:
            ax = axes[irow, icol]
            self._plotMetric(metric, metric_dct[metric], ax=ax, is_plot=False)
            if irow < nrow - 1:
                ax.set_xlabel("")
            icol += 1
            if icol >= ncol:
                icol = 0
                irow += 1
        if is_plot:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    analyzer = SensitivityAnalyzer()
    analyzer.makeData(nrml_stds=NRML_STDS, num_sample=NUM_SAMPLE)
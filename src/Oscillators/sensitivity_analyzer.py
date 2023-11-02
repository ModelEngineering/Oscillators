"""Analyzes the sensitivity of the 2SHO to variations in parameter values."""

import src.Oscillators.constants as cn
from src.Oscillators import util
from src.Oscillators.solver import Solver

import collections
import numpy as np
import os
import pandas as pd
import sympy as sp

X_TERMS = [cn.C_X1, cn.C_X2]
PARAMETERS = [p for p in cn.ALL_PARAMETERS if not p in [cn.C_T, cn.C_X1, cn.C_X2]]
NUM_SAMPLE = 1000

SENSITIVITY_DATA_DIR = os.path.join("%s", "sensitivity_data")
DEVIATION_DIR = os.path.join(SENSITIVITY_DATA_DIR, "%s")
MEAN_PATH = os.path.join(DEVIATION_DIR, "mean.csv")
STD_PATH = os.path.join(DEVIATION_DIR, "mean.csv")
OTHER_PATH = os.path.join(DEVIATION_DIR, "other.csv") 


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
        self.oc_df, self.baseline_oc_df = self.makeOscillationCharacteristicsDF()

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
        std = self.baseline_oc_df.loc[parameter_name, x_term]*cv
        result = np.random.normal(self.baseline_parameter_df.loc[parameter_name, x_term], std, num_sample)
        return result
    
    def _getRandomValues(self, parameter_name, frac_deviation, num_sample):
        """Returns a random value of the parameter"""
        lower = self.baseline_parameter_dct[parameter_name]*(1 - frac_deviation)
        upper = self.baseline_parameter_dct[parameter_name]*(1 + frac_deviation)
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
    
    def _makeRandomParameterDct(self, frac_deviation=1, num_sample=1000):
        """
        Creates random sample of parameter values sampling from a mean adjusted distribution.

        Args:
            frac_deviation: float (fractional deviation from the baseline value)
            num_sample: int (number of samples to generate)

        Returns:
            dict
                key: str (parameter name)
                value: np.ndarray (random sample of parameter values)
        """
        return {p: self._getRandomValues(p, frac_deviation, num_sample) for p in PARAMETERS}

    def makeErrorStatistics(self, frac_deviation=1, num_sample=NUM_SAMPLE):
        """
        Calculates the average absolute error of each OC for the coefficient of variation.

        Args:
            frac_deviation: float (fractional deviation from the baseline value)
        Returns:
            ErrorStatistic
                mean_df - mean absolute error for each OC
                std_df - std of absolute error for each OC
                num_negative - number of infeasible solutions
                num_nonoscillating - number of non-oscillating solutions
        """
        parameter_sample_dct = self._makeRandomParameterDct(frac_deviation=frac_deviation, num_sample=num_sample)
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
                    sample_value = float(sp.N(self.oc_df.loc[oc, x_term].subs(symbol_dct)))
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
                baseline_value = self.baseline_oc_df.loc[oc, x_term]
                abs_error_arr = np.abs((np.array(oc_sample_dct[x_term][oc]) - baseline_value)/baseline_value)
                mean_dct[x_term][oc] = np.mean(abs_error_arr)
                std_dct[x_term][oc] = np.std(abs_error_arr)
        # Construct the result
        mean_df = self._makeDataFrameFromTwoLevelDct(mean_dct)
        std_df = self._makeDataFrameFromTwoLevelDct(std_dct)
        frac_infeasible = (num_nonoscillating + num_negative)/num_sample
        return ErrorStatistic(mean_df=mean_df, std_df=std_df, frac_nonoscillating=num_negative/num_sample,
                              frac_infeasible=frac_infeasible, sample_size=num_sample)
    
    def makeData(self, frac_deviations, num_sample=NUM_SAMPLE, data_dir=cn.DATA_DIR):
        """
        Creates the data needed for plotting.

        Args:
            frac_deviations (_type_): _description_
            num_sample (_type_, optional): _description_. Defaults to NUM_SAMPLE.
        """
        cur_dir = SENSITIVITY_DATA_DIR % data_dir
        import pdb; pdb.set_trace()
        if not os.path.isdir(cur_dir):
            os.mkdir(cur_dir)
        for frac in frac_deviations:
            path_dir = {"mean": MEAN_PATH % (data_dir, str(frac)),
                        "std": STD_PATH % (data_dir, str(frac)),
                        "other": OTHER_PATH % (data_dir, str(frac))
                        }
            import pdb; pdb.set_trace()
            if not os.path.isdir(deviation_dir):
                os.mkdir(deviation_dir)
            mean_path = MEAN_PATH % frac
            std_path = STD_PATH % frac
            other_path = OTHER_PATH % frac
            statistics = self.makeErrorStatistics(frac_deviation=frac, num_sample=num_sample)
            statistics.mean_df.to_csv(mean_path)
            statistics.std_df.to_csv(std_path)
            other_df = pd.DataFrame({cn.C_NONOSCILLATING: [statistics.frac_nonoscillating],
                                     cn.C_INFEASIBLE: [statistics.frac_infeasible],
                                     cn.C_SAMPLE_SIZE: [statistics.sample_size]})
            other_df.to_csv(other_path)
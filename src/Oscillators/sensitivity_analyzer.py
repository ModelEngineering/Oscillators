"""Analyzes the sensitivity of the 2SHO to variations in parameter values."""

import src.Oscillators.constants as cn
from src.Oscillators import util
from src.Oscillators.solver import Solver

import collections
import numpy as np
import numpy.ma as ma
import pandas as pd
import sympy as sp

X_TERMS = [cn.C_X1, cn.C_X2]

ErrorStatistic = collections.namedtuple("ErrorStatistic", "mean_df std_df frac_infeasible frac_nonoscillating sample_size")


class SensitivityAnalyzer(object):

    def __init__(self, parameter_dct=None):
        """Constructor.

        Args:
            parameter_dct: (dict) keys are parameter names, values are parameter values
        """
        if parameter_dct is None:
            parameter_dct = dict(cn.PARAM_DCT)
            parameter_dct[cn.C_X1_0] = parameter_dct[cn.C_S1]
            parameter_dct[cn.C_X2_0] = parameter_dct[cn.C_S2]
            del parameter_dct[cn.C_S1]
            del parameter_dct[cn.C_S2]
        new_parameter_dct = dict(parameter_dct)
        Solver.calculateDependentParameters(new_parameter_dct)
        self.solver = Solver()
        self.solver.solve()
        self.symbol_dct = util.makeSymbolDct(self.solver.x_vec, new_parameter_dct)
        self.oc_df, self.baseline_df = self.makeBaselineDF()

    def makeBaselineDF(self):
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
        # Calculate the baseline values for the oscillation characteristics
        for idx in oc_df.index:
            for column in oc_df.columns:
                value = float(sp.N(oc_df.loc[idx, column].subs(self.symbol_dct)))
                baseline_df.loc[idx, column] = value
        return oc_df.T, baseline_df.T
    
    def _getRandomValues(self, x_term, parameter_name, cv, num_sample):
        """Returns a random value of the parameter"""
        std = self.baseline_df.loc[parameter_name, x_term]*cv
        return np.random.normal(self.baseline_df.loc[parameter_name, x_term], std, num_sample)

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
    
    def _makeRandomParameterDct(self, cv=1, num_sample=1000):
        """
        Creates random sample of parameter values sampling from a mean adjusted distribution.

        Args:
            cv: float (coefficient of variation for the distribution of parameter values)
            num_sample: int (number of samples to generate)

        Returns:
            dict (two level)
        """
        parameter_sample_dct = self._initializeTwoLevelDct()
        for x_term in X_TERMS:
            for oc in cn.OSCILLATION_CHARACTERISTICS:
                parameter_sample_dct[x_term][oc] = self._getRandomValues(oc, x_term, cv, num_sample)
        return parameter_sample_dct

    def calculateErrorStatistics(self, cv=1, num_sample=1000):
        """
        Calculates the average absolute error of each OC for the coefficient of variation.

        Args:
            cv: float (coefficient of variation for the distribution of parameter values)
        Returns:
            ErrorStatistic
                mean_df - mean absolute error for each OC
                std_df - std of absolute error for each OC
                num_infeasible - number of infeasible solutions
                num_nonoscillating - number of non-oscillating solutions
        """
        parameter_sample_dct = self._makeRandomParameterDct(cv=1, num_sample=1000)
        # Obtain samples of oscillation characteristics from the sampled parameter values
        oc_sample_dct = self._initializeTwoLevelDct()
        num_nonoscillating = 0
        for idx in num_sample:
            for x_term in X_TERMS:
                parameter_dct = {oc: parameter_sample_dct[x_term][oc][idx] for oc in cn.OSCILLATION_CHARACTERISTICS}
                symbol_dct = util.makeSymbolDct(self.solver.x_vec, parameter_dct)
                # Correct for "titration" experiment
                parameter_dct[cn.C_K3] = parameter_dct[cn.C_K1] + parameter_dct[cn.C_K2]
                # Check for negative concentrations
                if parameter_dct[cn.C_k3] > parameter_dct[cn.C_K5]:
                    num_nonoscillating += 1
                    continue
                for oc in cn.OSCILLATION_CHARACTERISTICS:
                    # Calculate the oscillation characteristic
                    sample_value = float(sp.N(self.oc_df.loc[x_term, oc].subs(symbol_dct)))
                    oc_sample_dct[x_term][oc].append(sample_value)
        # Calculate instances of negative concentrations, which are infeasible
        num_infeasible = 0
        for x_term in X_TERMS:
            amplitude_arr = np.array(oc_sample_dct[x_term][cn.C_ALPHA])
            omega_arr = np.array(oc_sample_dct[x_term][cn.C_OMEGA])
            num_infeasible += sum(amplitude_arr > omega_arr)
        # Calculate the average absolute error
        mean_dct = self._initializeTwoLevelDct()
        std_dct = self._initializeTwoLevelDct()
        for x_term in X_TERMS:
            for oc in cn.OSCILLATION_CHARACTERISTICS:
                baseline_value = self.baseline_df.loc[x_term, oc]
                abs_error_arr = (ma.array(oc_sample_dct[x_term][oc]) - baseline_value)/baseline_value
                mean_dct[x_term][oc] = np.mean(abs_error_arr)
                std_dct[x_term][oc] = np.std(abs_error_arr)
        # Construct the result
        mean_df = self._makeDataFrameFromTwoLevelDct(mean_dct)
        std_df = self._makeDataFrameFromTwoLevelDct(std_dct)
        return ErrorStatistic(mean_df=mean_df, std_df=std_df, frac_infeasible=num_infeasible/num_sample,
                              frac_nonoscillating=num_nonoscillating/num_sample, sample_size=num_sample)
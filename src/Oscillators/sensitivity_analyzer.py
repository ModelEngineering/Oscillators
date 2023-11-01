"""Analyzes the sensitivity of the 2SHO to variations in parameter values."""

import src.Oscillators.constants as cn
from src.Oscillators import util
from src.Oscillators.solver import Solver

import numpy as np
import sympy as sp


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
        oc_df = self.solver.calculateOscillationCharacteristics()
        self.symbol_dct = util.makeSymbolDct(self.solver.x_vec, new_parameter_dct)
        self.baseline_df = oc_df.copy()
        # Calculate baseline values for oscillation characteristics
        for idx in oc_df.index:
            for column in oc_df.columns:
                value = float(sp.N(oc_df.loc[idx, column].subs(self.symbol_dct)))
                self.baseline_df.loc[idx, column] = value

    def calculateSensitivity(self, cv=1):
        """
        Calculates the average rms error and its standard deviation for the cv using a monte carlo approach.

        Args:
            cv: float (coefficient of variation for the distribution of parameter values)
        Returns:
            pd.DataFrame
                columns: C_X1, C_X2
                index: C_THETA, C_ALPHA, C_PHI, C_OMEGA
        """
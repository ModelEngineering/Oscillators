'''Constructs a closed form solution to the ODE for the Oscillator'''

from src.Oscillators import c1, c2, t, \
      theta, k2, k4, k6, k_d, x1_0, x2_0, I
from src.Oscillators.static_expressions import xp_1, xp_2
from src.Oscillators import util
import src.Oscillators.constants as cn

import collections
import numpy as np
import pandas as pd
import tellurium as te
import sympy as sp
import matplotlib.pyplot as plt


A = "a"
B = "b"
C = "c"

OscillatorCharacteristics = collections.namedtuple("OscillatorCharacteristics", "theta alpha phi omega")



class Solver(object):

    # This class derivces the time domain solution

    def __init__(self):
        self.u_vec = sp.Matrix([[-k4], [k6]])
        self.A_mat = sp.Matrix([[k2, k2], [-k2 - k_d, -k2]])
        w_vec = sp.Matrix([ [-k2/(k2 + k_d) - theta*I/(k_d + k2)], [1]])*(sp.exp(I*theta*t))
        vecs = w_vec.as_real_imag()
        # The fundamental matrix
        self.fund_mat = sp.Matrix([ [vecs[0][0], vecs[1][0]], [vecs[0][1], vecs[1][1]]])
        # The following are calculated by solve
        self.homogeneous_x_vec = None # Homogeneous solution
        self.particular_x_vec = None # Particular solution
        self.raw_x_vec = None # Initial time domain solution
        self.factor_dcts = None # Dictionaries of the sinusoid coefficients
        self.factored_x_vec = None # Time domain solution factored into sinusoids
        self.theta = self.calculateTheta(k2, k_d)
        self.alphas = None # Amplitudes of the sinusoids
        self.phis = None # Phases of the sinusoids 
        self.omegas = None # Oscillation offsets
        self.x_vec = None # Solution structured as a sine with a phase shift

    @staticmethod
    def calculateTheta(kk2, kkd):
        if "symbol" in str(type(kk2)):
            return sp.sqrt(kk2*kkd)
        else:
            return sp.sqrt(kk2*kkd)
        
    @staticmethod
    def calculateDependentParameters(dct):
        """
        Calculates the dependent parameters.

        Args:
            dct (dict): key: str, value: float
        """
        dct[cn.C_K1] = cn.K1_VALUE
        dct[cn.C_K3] = dct[cn.C_K1] + dct[cn.C_K2]
        dct[cn.C_K5] = dct[cn.C_K3] + dct[cn.C_K_D]
        dct[cn.C_THETA] = np.sqrt(dct[cn.C_K2]*dct[cn.C_K_D])
        dct[cn.C_S1] = dct[cn.C_X1_0]
        dct[cn.C_S2] = dct[cn.C_X2_0]

    def getOscillatorCharacteristics(self, dct=None):
        """Returns a substituted value.

        Args:
            dct: dict (key: name, value: float)

        Returns:
            OscillatorCharacteristics (x1)
            OscillatorCharacteristics (x2)
        """
        if dct is None:
            dct = {}
        def get(idx):
            return OscillatorCharacteristics(
                theta=util.getSubstitutedExpression(self.theta, dct),
                alpha=util.getSubstitutedExpression(self.alphas[idx], dct),
                phi=util.getSubstitutedExpression(self.phis[idx], dct),
                omega=util.getSubstitutedExpression(self.omegas[idx], dct)
            )
        return get(0), get(1)

    def solve(self, is_check=False, is_simplify=False):
        """
        Constructs the time domain solution.

        Args:
            is_check (bool, optional): check the saved solution. Defaults to False.
            is_simplify: simplify the final solution (time consuming)
        Returns:
            sp.Matrix: matrix solution for the differential equation
        """
        # Calculate the homogeneous solution
        self.homogeneous_x_vec = self.fund_mat*sp.Matrix([[c1], [c2]])
        xpp_1 = eval(xp_1)
        xpp_2 = eval(xp_2)
        self.particular_x_vec = sp.Matrix([[xpp_1], [xpp_2]])
        if is_check:
            # Calculate the particular solution
            rhs = sp.simplify(self.fund_mat.inv()*self.u_vec)
            rhs = sp.integrate(rhs, t)
            xp = sp.simplify(self.fund_mat*rhs)
            if not xp == self.particular_x_vec:
                raise RuntimeError("Saved and calculated particular solutions do not match!")
        # Solve for the constants in terms of the initial conditions
        self.raw_x_vec = self.homogeneous_x_vec + self.particular_x_vec
        cdct = sp.solve(self.raw_x_vec.subs(t, 0) - sp.Matrix([ [x1_0], [x2_0]]), [c1, c2])
        self.raw_x_vec = self.raw_x_vec.subs(cdct)
        # Factor the solution into polynomials of sine and cosine
        self.factor_dcts = []
        factored_x_vec = []
        for xterm in self.raw_x_vec:
            dct = self._findSinusoidCoefficients(xterm)
            self.factor_dcts.append(dct)
            factored_term = dct[A]*sp.cos(t*theta) + dct[B]*sp.sin(t*theta) + dct[C]
            factored_x_vec.append(factored_term)
        self.factored_x_vec = sp.Matrix(factored_x_vec)
        # Create a solution in terms of only sine
        x_terms = []
        self.alphas = []
        self.phis = []
        self.omegas = []
        for dct in self.factor_dcts:
            amplitude = sp.simplify(sp.sqrt(dct[A]**2 + dct[B]**2))
            phase = sp.simplify(sp.atan(dct[A]/dct[B]))
            phase = sp.Piecewise((phase, dct[B] >= 0), (phase + sp.pi, dct[B] < 0))
            x_term = amplitude*sp.sin(t*theta + phase) + dct[C]
            x_terms.append(x_term)
            self.alphas.append(amplitude)
            self.phis.append(phase)
            self.omegas.append(dct[C])
        self.x_vec = sp.Matrix(x_terms)
        if is_simplify:
            self.x_vec = sp.simplify(self.x_vec)
        return self.x_vec

    @staticmethod
    def _findSinusoidCoefficients(expression):
        """
        Finds the coefficients of the sinusoids.

        Args:
            expression: sp.expression
        Assumes:
            The symbols t, theta are defined symbols
        Returns:
            dct
                key: "a" (coefficient of cosine), "b" (coefficient of sine), "c" (constant coefficient)
                value: expression
        """
        # Construct dictionaries for the sinusoid terms
        sin_dct = util.makePolynomialCoefficients(expression, sp.sin(t*theta))
        cosine_dct = util.makePolynomialCoefficients(sin_dct[0], sp.cos(t*theta))
        # Create the result
        result_dct = dict(cosine_dct)
        result_dct[A] = result_dct[1].as_expr()
        del result_dct[1]
        result_dct[B] = sin_dct[1].as_expr()
        result_dct[C] = result_dct[0].as_expr()
        del result_dct[0]
        return result_dct

    def simulate(self, expression=None, param_dct=cn.PARAM_DCT, end_time=5, **kwargs):
        """
        Simulates an expression over time.

        Args:
            expression (sp.Matrix, optional): expression to simulate. Defaults to self.x_vec
            param_dct: dict
                key: str
                value: float
            kwargs: dict
                keyword arguments for util.plotDF
        Returns:
            pd.DataFrame
                key: time
                Columns: species names
        """
        if expression is None:
            if self.x_vec is None:
                raise ValueError("expression is None and self.x_vec is None")   
            expression = self.x_vec
        new_param_dct = dict(param_dct)
        new_param_dct["theta"] = self.calculateTheta(param_dct["k2"], param_dct["k_d"])
        symbol_dct = util.makeSymbolDct(expression, new_param_dct)
        if expression == self.A_mat:
            mat = self.A_mat.subs(symbol_dct)
            df = util.simulateLinearSystem(A=mat, end_time=end_time, column_names=["S1", "S2"], **kwargs)
        else:
            vector_func = expression.subs(symbol_dct)
            df = util.simulateExpressionVector(vector_func, new_param_dct, end_time=end_time, **kwargs)
        return df
    
    def _parameterToStr(self, parameter_name, parameter_value, num_digits=4):
        stg = "%s=%s" % (parameter_name, parameter_name)
        return stg
    
    def plotFit(self, ax=None, is_plot=True, output_path=None, xlabel="time", title=None,
                param_dct=cn.PARAM_DCT, end_time=5,
                 is_legend=True, is_xaxis=True):
        """
        Plots the fit of the mathematical model to the simulation for the specified values of the parameters.

        Args:
            ax: matplotlib.Axes
            is_plot: bool
            output_path: str
            xlabel: str
            title: str
            param_dct: dict (parameters)
                key: str
                value: float
            end_time: float (end time for simulation)
            is_legend: bool (include the legend)
            is_xaxis: bool (include the x-axis)
        """
        if ax is None:
            _, ax = plt.subplots()
        if title is None:
            stgs = [self._parameterToStr(k,v) for k, v in  param_dct.items()]
            title = ", ".join(stgs)
        # Calulate the values
        self.calculateDependentParameters(param_dct)
        prediction_df = self.simulate(expression=self.x_vec, param_dct=param_dct, end_time=end_time, is_plot=False)
        simulation_df = util.simulateRR(param_dct=param_dct, end_time=end_time, is_plot=False, num_point=len(prediction_df))
        times = simulation_df.index.to_list()
        # Plot
        ax.set_title(title, fontsize=10)
        # FIXME: better colors
        for color, species in zip(["black", "blue"], ["S1", "S2"]):
            ax.plot(times, simulation_df[species], label="True", color=color)
            ax.scatter(times, prediction_df[species], label="Predicted", color=color, marker="*")
        if is_xaxis:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])
        if is_legend:
            ax.legend(["True S1", "Predicted S1", "True S2", "Predicted S2"])
        if output_path is not None:
            ax.figure.savefig(output_path)
        if is_plot:
            plt.show()
        else:
            plt.close()

    def plotManyFits(self, output_path=None, is_plot=True):
        """
        Constructs a grid of plots for the different parameters.

        Args:
            output_path: str
            is_plot: bool
        """
        END_TIME = 2
        dct = {}
        dct["k2"] = [11.35, 5.97, 9.78, 16.81]
        dct["k_d"] = [2.20, 4.18, 10, 23, 5.94]
        dct["k4"] = [118.82, 372.57, 119.92, 777.43]
        dct["k6"] = [129.83, 592.03, 169.95, 993.03]
        dct["x1_0"] = [5.0, 57.33, 5.0, 40.64]
        dct["x2_0"] = [7.66, 10.0, 2.038, 10.0]
        length = len(dct["k2"])
        nrow = length//2
        ncol = nrow
        fig, axs = plt.subplots(nrow, ncol) 
        irow = 0
        icol = 0
        for idx in range(length):
            if irow == nrow - 1:
                is_xaxis = True
            else:
                is_xaxis = False
            param_dct = {n: v[idx] for n, v in dct.items()}
            self.plotFit(is_plot=False, param_dct=param_dct, ax=axs[irow, icol], is_legend=False, is_xaxis=is_xaxis,
                          end_time=END_TIME)
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
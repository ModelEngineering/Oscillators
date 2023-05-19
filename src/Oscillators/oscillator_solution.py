'''Constructs a closed form solution to the ODE for the Oscillator'''

from src.Oscillators import c1, c2, t, \
      alpha, omega, theta, k2, k4, k6, k_d, x1, x1_0, x2, x2_0, I
from src.Oscillators.static_expressions import xp_1, xp_2
from src.Oscillators import util

import tellurium as te
import sympy as sp
from sympy import init_printing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

A = "a"
B = "b"
C = "c"


class OscillatorSolution(object):

    # This class derivces the time domain solution

    def __init__(self):
        self.u_vec = sp.Matrix([[-k4], [k6]])
        self.A_mat = sp.Matrix([[k2, k2], [-k2 - k_d, -k2]])
        w_vec = sp.Matrix([ [-k2/(k2 + k_d) - theta*I/(k_d + k2)], [1]])*(sp.exp(I*theta*t))
        vecs = w_vec.as_real_imag()
        self.fund_mat = sp.Matrix([ [vecs[0][0], vecs[1][0]], [vecs[0][1], vecs[1][1]]])
        self.raw_x_vec = None # Initial time domain solution
        self.factored_x_vec = None # Time domain solution factored into sinusoids
        self.x_vec = None # Solution structured as a sine with a phase shift

    def getSolution(self, is_check=False):
        """
        Constructs the time domain solution.

        Args:
            is_check (bool, optional): check the saved solution. Defaults to False.
        Returns:
            sp.Matrix: matrix solution for the differential equation
        """
        # Calculate the homogeneous solution
        xhh = self.fund_mat*sp.Matrix([[c1], [c2]])
        xpp_1 = eval(xp_1)
        xpp_2 = eval(xp_2)
        xp_saved = sp.Matrix([[xpp_1], [xpp_2]])
        if is_check:
            # Calculate the particular solution
            rhs = sp.simplify(self.fund_mat.inv()*self.u_vec)
            rhs = sp.integrate(rhs, t)
            xp = sp.simplify(self.fund_mat*rhs)
            if not xp == xp_saved:
                raise RuntimeError("Saved and calculated particular solutions do not match!")
        # Solve for the constants in terms of the initial conditions
        raw_x_vec = xhh + xp_saved
        cdct = sp.solve(raw_x_vec.subs(t, 0) - sp.Matrix([ [x1_0], [x2_0]]), [c1, c2])
        self.raw_x_vec = raw_x_vec.subs(cdct)
        # Factor the solution into polynomials of sine and cosine
        factored_x_vec = []
        factor_dcts = []
        for xterm in self.raw_x_vec:
            dct = self._findSinusoidCoefficients(xterm)
            factor_dcts.append(dct)
            factored_term = dct[A]*sp.cos(t*theta) + dct[B]*sp.sin(t*theta) + dct[C]
            factored_x_vec.append(factored_term)
        self.factored_x_vec = sp.Matrix(factored_x_vec)
        # Create a solution in terms of only sine
        x_terms = []
        for dct in factor_dcts:
            amplitude = sp.simplify(sp.sqrt(dct[A]**2 + dct[B]**2))
            phase = sp.simplify(sp.atan(dct[A]/dct[B]))
            x_term = amplitude*sp.sin(t*theta + phase) + dct[C]
            x_terms.append(x_term)
        self.x_vec = sp.Matrix(x_terms)
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
        
        def makeDct(expr, term):
            """
            Creates a dictionary that has the coefficients of the term
            """
            tt = sp.Symbol("tt")
            new_expr = expr.subs({term: tt})
            collected = sp.Poly(new_expr, tt).as_expr()
            i, d = collected.as_independent(tt, as_Add=True)
            rv = dict(i.as_independent(tt, as_Mul=True)[::-1] for i in sp.Add.make_args(d))
            if i:
                assert 1 not in rv
                rv.update({sp.S.One: i})
            rv[0] = rv[1].as_expr()
            del rv[1]
            rv[1] = rv[tt].as_expr()
            del rv[tt]
            return rv
        #
        # Make the sine dictionary
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
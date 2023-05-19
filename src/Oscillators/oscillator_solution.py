'''Constructs a closed form solution to the ODE for the Oscillator'''

from src.Oscillators import c1, c2, T, phi, r1, r2, t, b1, b2, m1, m2, h1, h2, \
      alpha, omega, theta, S1, S2, k1, k2, k3, k4, k5, k6, k_d, x1, x1_0, x2, x2_0, I
from src.Oscillators.static_expressions import xp_1, xp_2

import tellurium as te
import sympy as sp
from sympy import init_printing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import control
import lmfit


class OscillatorSolution(object):

    # This class derivces the time domain solution

    def __init__(self):
        self.u_vec = sp.Matrix([[-k4], [k6]])
        self.A_mat = sp.Matrix([[k2, k2], [-k2 - k_d, -k2]])
        w_vec = sp.Matrix([ [-k2/(k2 + k_d) - theta*I/(k_d + k2)], [1]])*(sp.exp(I*theta*t))
        vecs = w_vec.as_real_imag()
        self.fund_mat = sp.Matrix([ [vecs[0][0], vecs[1][0]], [vecs[0][1], vecs[1][1]]])
        self.x_vec = None

    def getSolution(self, is_check=False):
        """Constructs the time domain solution.

        Args:
            is_check (bool, optional): check the saved solution. Defaults to False.

        Returns:
            sympy.Matrix: matrix solution for the differential equation
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
        #
        self.x_vec = xhh + xp_saved
        return self.x_vec
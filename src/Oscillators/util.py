from src.Oscillators.model import MODEL, PARAM_DCT 
from src.Oscillators import t, theta, k_d, k1, k2, k3, k4, k5, k6

import tellurium as te
import sympy
from sympy import init_printing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import control
import lmfit


def makeTimes(start_time=0, end_time=5.0, point_density=20):
    num_point = int(point_density*(end_time - start_time))
    return np.linspace(start_time, end_time, num_point)

TIMES = makeTimes()

def simulateLinearSystem(A=None, B=None, end_time=20, is_plot=True):
    """
    Simulates the linear system specified by A and B
    """
    if A is None:
        A = np.array([ [PARAM_DCT["k1"] - PARAM_DCT["k2"], PARAM_DCT["k4"]],
                           [PARAM_DCT["k2"] - PARAM_DCT["k3"], - PARAM_DCT["k4"] ] ])
    if B is None:
        B = np.eye(2)
    C = np.eye(2)
    D = 0*np.eye(2)
    sys = control.StateSpace(A, B, C, D)
    sys = control.LinearIOSystem(sys, inputs=["S1", "S2"], outputs=["S1", "S2"])
    X0 = [1, 10]
    times = makeTimes(end_time=end_time)
    response = control.input_output_response(sys, T=times, X0=X0)
    plt.plot(response.t, response.y[0])
    plt.plot(response.t, response.y[1])
    plt.legend(["S1", "S2"], loc="upper left")
    if not is_plot:
        plt.close()

def simulateExpression(sym, dct, times=TIMES, is_plot=True):
    """
    Simulates a symbol that is a function of time.
    The time symbol must be "t".

    Parameters
    ----------
    sym: sympy.Symbol
    t: sympy.Symbol
    dct: dict (substitutions)

    Returns
    -------
    list-float
    """
    # Find the time symbol
    time_syms = [a for a in sym.atoms() if str(a) == "t"]
    if len(time_syms) == 0:
        raise ValueError("No time found!")
    if len(time_syms) > 1:
        raise ValueError("Multiple times found!")
    t = time_syms[0]
    # Simulation of ivp_solution
    new_sym = sym.subs(dct)
    vals = [float(sympy.simplify(new_sym.subs(t, v))) for v in times]
    if is_plot:
        plt.plot(times, vals)
    return vals

def simulateExpressionVector(vec, dct, end_time=round(TIMES[-1]), is_plot=True):
    """
    Simulates a 2-d vector symbol that is a function of time.
    The time symbol must be "t".

    Parameters
    ----------
    sym: sympy.Symbol
    dct: dict (substitutions)
    """
    times = makeTimes(end_time=end_time)
    s1_vals = simulateExpression(vec[0], dct, times=times)
    s2_vals = simulateExpression(vec[1], dct, times=times)
    #
    plt.plot(times, s1_vals)
    plt.plot(times, s2_vals)
    _ = plt.legend(["S1", "S2"])
    if not is_plot:
        plt.close()

def simulateRR(dct={}, is_plot=True):
    """
    Simulates the model with parameter updates as indicated.
    """
    rr = te.loada(MODEL)
    for key, value in PARAM_DCT.items():
        if key in rr.keys():
            rr[key] = value
    for key, value in dct.items():
        if key in rr.keys():
            rr[key] = value
    data = rr.simulate(0, 5, 1000)
    for col in data.colnames[1:]:
        plt.plot(data[:, 0], data[col])
    plt.legend(data.colnames[1:])
    if not is_plot:
        plt.close()

def makeSymbolDct(expression, name_dct, exclude_names=None):
    """
    Creates a dictionary whose keys are symbols in the expression with the same name.
    """
    if exclude_names is None:
        exclude_names = []
    convert_dct = {s.n: s for s in expression.free_symbols}
    dct = {}
    for sym in convert_dct.values():
        if sym.name in exclude_names:
            continue
        if not sym.name in name_dct.keys():
            continue
        dct[sym] = name_dct[sym.name]
    #
    return dct

def findSinusoidCoefficients(expression):
    """
    Finds the coefficients of the sinusoids.

    Args:
        expression: sympy.expression
    Assumes:
        The symbols t, theta are defined symbols
    Returns:
        dct
            key: "a" (coefficient of cosine), "b" (coefficient of sine), "c" (constant coefficient)
            value: expression
    """
    A = "a"
    B = "b"
    C = "c"
    def makeDct(expr, term):
        """
        Creates a dictionary that has the coefficients of the term
        """
        tt = sympy.Symbol("tt")
        new_expr = expr.subs({term: tt})
        collected = sympy.Poly(new_expr, tt).as_expr()
        i, d = collected.as_independent(tt, as_Add=True)
        rv = dict(i.as_independent(tt, as_Mul=True)[::-1] for i in sympy.Add.make_args(d))
        if i:
            assert 1 not in rv
            rv.update({sympy.S.One: i})
        rv[0] = rv[1].as_expr()
        del rv[1]
        rv[1] = rv[tt].as_expr()
        del rv[tt]
        return rv
    #
    # Make the sine dictionary
    sin_dct = makeDct(expression, sympy.sin(t*theta))
    cosine_dct = makeDct(sin_dct[0], sympy.cos(t*theta))
    # Create the result
    result_dct = dict(cosine_dct)
    result_dct[A] = result_dct[1].as_expr()
    del result_dct[1]
    result_dct[B] = sin_dct[1].as_expr()
    result_dct[C] = result_dct[0].as_expr()
    del result_dct[0]
    return result_dct
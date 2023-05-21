from src.Oscillators.model import MODEL, PARAM_DCT 
from src.Oscillators import t, theta

import tellurium as te
import sympy as sp
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

def plotDF(df, is_plot=True, output_path=None, title="", xlabel="time", ylabel="value", xlim=None, ylim=None):
    """Plots the dataframe

    Args:
        df: pd.DataFrame
            index: time
            columns: column_names
        is_plot (bool, optional): _description_. Defaults to True.
        output_path (str, optional): path to the output file
    """
    ax = df.plot()
    plt.legend(df.columns, loc="upper left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if output_path is not None:
        ax.figure.savefig(output_path)
    if is_plot:
        plt.show()

def simulateLinearSystem(A=None, B=None, end_time=20, column_names=None, **kwargs):
    """
    Simulates the linear system specified by A and B

    Args:
        A, B: matrices of the linear system
        end_time: float (end time of simulation)
        column_names: list-str (names of the columns)
        kwargs: dict (arguments to plotDF)

    Returns:
        pd.DataFrame
            key: time
            columns: column_names
    """
    if column_names is None:
        column_names = ["S1", "S2"]
    if A is None:
        A = np.array([ [PARAM_DCT["k2"], PARAM_DCT["k2"]],
                           [-PARAM_DCT["k2"] - PARAM_DCT["k_d"], - PARAM_DCT["k2"] ] ])
    if B is None:
        B = np.eye(2)
    C = np.eye(2)
    D = 0*np.eye(2)
    sys = control.StateSpace(A, B, C, D)
    sys = control.LinearIOSystem(sys, inputs=column_names, outputs=column_names)
    X0 = [1, 10]
    times = makeTimes(end_time=end_time)
    response = control.input_output_response(sys, T=times, X0=X0)
    dct = {"time": response.t}
    for idx, name in enumerate(column_names):
        dct[name] = response.y[idx]
    df = pd.DataFrame(dct)
    df = df.set_index("time")
    plotDF(df, **kwargs)
    return df

def simulateExpression(sym, dct, times=TIMES, **kwargs):
    """
    Simulates a symbol that is a function of time.
    The time symbol must be "t".

    Args:
        sym: sp.Symbol
        t: sp.Symbol
        dct: dict (substitutions)
        kwargs: dict (arguments to plotDF)

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
    vals = [float(sp.simplify(new_sym.subs(t, v))) for v in times]
    df = pd.DataFrame({"time": times, "value": vals})
    df = df.set_index("time")
    plotDF(df, **kwargs)
    return vals

def simulateExpressionVector(vec, dct, end_time=round(TIMES[-1]), column_names=None, **kwargs):
    """
    Simulates a 2-d vector symbol that is a function of time.
    The time symbol must be "t".

    Parameters
    ----------
    sym: sp.Symbol
    dct: dict (substitutions)

    Returns
    -------
    pd.DataFrame
        key: time
    """
    if column_names is None:
        column_names = ["S1", "S2"]
    times = makeTimes(end_time=end_time)
    s1_vals = simulateExpression(vec[0], dct, times=times)
    s2_vals = simulateExpression(vec[1], dct, times=times)
    df = pd.DataFrame({"time": times, column_names[0]: s1_vals, column_names[1]: s2_vals})
    df = df.set_index("time")
    #
    plotDF(df, **kwargs)
    return df

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
        expression: sp.expression
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
    sin_dct = makeDct(expression, sp.sin(t*theta))
    cosine_dct = makeDct(sin_dct[0], sp.cos(t*theta))
    # Create the result
    result_dct = dict(cosine_dct)
    result_dct[A] = result_dct[1].as_expr()
    del result_dct[1]
    result_dct[B] = sin_dct[1].as_expr()
    result_dct[C] = result_dct[0].as_expr()
    del result_dct[0]
    return result_dct

def makePolynomialCoefficients(expression, term):
    """
    Creates a dictionary with the polynomial coefficients of the term.

    Args:
        expression: sp.Expression
        term: sp.Expression or sp.Symbol

    Returns:
        dict
            key: int (power of term)
            value: sp.Expression
    """
    tt = sp.Symbol("tt")
    new_expr = expression.subs({term: tt})
    collected = sp.Poly(new_expr, tt).as_expr()
    i, d = collected.as_independent(tt, as_Add=True)
    rv = dict(i.as_independent(tt, as_Mul=True)[::-1] for i in sp.Add.make_args(d))
    if i:
        assert 1 not in rv
        rv.update({sp.S.One: i})
    dct = {}
    for key in range(len(rv.keys())):
        dct[key] = rv[tt**(key)]
    return dct
from src.Oscillators.constants import PARAM_DCT
from src.Oscillators.model import MODEL
from src.Oscillators import t, theta
import src.Oscillators.constants as cn

import tellurium as te
import sympy as sp
from sympy import init_printing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import control


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
    if not is_plot and output_path is None:
        return
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

def simulateExpressionVector(vec, dct, end_time=round(TIMES[-1]), times=None, column_names=None, **kwargs):
    """
    Simulates a 2-d vector symbol that is a function of time.
    The time symbol must be "t".

    Parameters
    ----------
    sym: sp.Symbol
    dct: dict (substitutions)
    times: list-float

    Returns
    -------
    pd.DataFrame
        key: time
    """
    if column_names is None:
        column_names = ["S1", "S2"]
    if times is None:
        times = makeTimes(end_time=end_time)
    s1_vals = simulateExpression(vec[0], dct, times=times, is_plot=False)
    s2_vals = simulateExpression(vec[1], dct, times=times, is_plot=False)
    df = pd.DataFrame({"time": times, column_names[0]: s1_vals, column_names[1]: s2_vals})
    df = df.set_index("time")
    #
    plotDF(df, **kwargs)
    return df

def simulateRR(param_dct={}, end_time=5, num_point=None, **kwargs):
    """
    Simulates the model with parameter updates as indicated.

    Args:
        param_dct: dict (parameter updates)
        end_time: float (end time of simulation)
    Returns:
        pd.DataFrame
    """
    rr = te.loada(MODEL)
    for key, value in PARAM_DCT.items():
        if key in rr.keys():
            rr[key] = value
    for key, value in param_dct.items():
        if key in rr.keys():
            rr[key] = value
    if num_point is None:
        num_point = int(10*end_time)
    data = rr.simulate(0, end_time, num_point)
    dct = {n: data[n] for n in data.colnames}
    df = pd.DataFrame(dct)
    df = df.set_index("time")
    df = df.rename(columns={s: s[1:-1] for s in df.columns})
    plotDF(df, **kwargs)
    return df

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

def getSubstitutedExpression(expression, name_dct, **kwargs):
    """Substitutes the symbols in the expression with the values in the dictionary.

    Args:
        expression: sp.Expression
        name_dct: dict (key: str, value: float)
        kwargs: optional arguments to makeSymbolDct
    """
    symbol_dct = makeSymbolDct(expression, name_dct, **kwargs)
    result = expression.subs(symbol_dct)
    if len(result.free_symbols) == 0:
        result = float(result)
    return result

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

def makeRandomParameterDct(min_val=0.1, max_val=1, cv=1, means=1, is_t=False, is_uniform=True):
    """
    Make a consistent set of parameter values.

    Args:
        min_val: float
        max_val: float
        cv: float (coefficient of variation)
        means: float/dict (mean of the parameter values)
            key: str (parameter name)
            value: float (parameter mean)
        is_t: bool (include the time parameter)
        is_uniform: bool (use a uniform distribution)

    Returns:
        dict
            key: str (parameter name)
            value: float (parameter value)
    """
    PARAMETERS = [cn.C_K1, cn.C_K2, cn.C_K3, cn.C_K4, cn.C_K_D,
           cn.C_K5, cn.C_K6, cn.C_X1_0, cn.C_X2_0, cn.C_THETA, "t"]
    if isinstance(means, float):
        mean_dct = {p: means for p in PARAMETERS}
    else:
        mean_dct = means
    def get(name):
        # Returns a random value for the named parameter
        if is_uniform:
            return np.random.uniform(min_val, max_val)
        else:
            return np.random.normal(mean_dct[name], cv*mean_dct[name])
    #
    k1 = get("k1")
    k2 = get("k2")
    k3 = k1 + k2
    k4 = get("k3")
    k_d = get("k4")
    k5 = k3 + k_d
    k6 = get("k6")
    x1_0 = get("x1_0")
    x2_0 = get("k2_0")
    theta = np.sqrt(k2*k_d)
    dct = {cn.C_K1: k1, cn.C_K2: k2, cn.C_K3: k3, cn.C_K4: k4, cn.C_K_D: k_d,
           cn.C_K5: k5, cn.C_K6: k6, cn.C_X1_0: x1_0, cn.C_X2_0: x2_0, cn.C_THETA: theta}
    if is_t:
        dct["t"] = get("t")
    return dct

def makeUniformRandomParameterDct(min_val=0.1, max_val=1, parameters=cn.ALL_PARAMETERS, is_calculate_dependent_parameters=True):
    """
    Make a consistent set of parameter values for values that are uniformly distributed.

    Args:
        min_val: float
        max_val: float
        parameters: list-str
        is_calculate_dependent_parameters: bool (calculate the dependent parameters)

    Returns:
        dict (parameter value dictionary)
            key: str (parameter name)
            value: float (parameter value)
    """
    def get(name):
        # Returns a random value for the named parameter
            return np.random.uniform(min_val, max_val)
    return _assignRandomParameterValues(get, parameters=parameters,
                                        is_calculate_dependent_parameters=is_calculate_dependent_parameters)

def makeNormalRandomParameterDct(cv=1, means=1, parameters=cn.ALL_PARAMETERS, is_calculate_dependent_parameters=True):
    """
    Make a consistent set of parameter values for values that are normally distributed.

    Args:
        cv: float (coefficient of variation)
        means: float/dict (mean of the parameter values)
            key: str (parameter name)
            value: float (parameter mean)
        parameters: list of parameters for which values are assigned
        is_calculate_dependent_parameters: bool (calculate the dependent parameters)

    Returns:
        dict (parameter value dictionary)
            key: str (parameter name)
            value: float (parameter value)
    """
    if isinstance(means, float):
        mean_dct = {p: means for p in cn.INDEPENDENT_PARAMETERS}
    else:
        mean_dct = means
    def get(name):
        if not name in mean_dct.keys():
            raise ValueError("Parameter not found: {}".format(name))
        return np.random.normal(mean_dct[name], cv*mean_dct[name])
    #
    return _assignRandomParameterValues(get, parameters=parameters,
                                        is_calculate_dependent_parameters=is_calculate_dependent_parameters)

def _assignRandomParameterValues(assignment_function, parameters=cn.ALL_PARAMETERS,
                                 is_calculate_dependent_parameters=True):
    """
    Assigns parameter values based on a random value assignment function.

    Args:
        assignment_function: function
            positional arguments: str (name of parameter)
            returns: float
        is_t: bool (include the time parameter)
        parameters: list of parameters for which values are assigned
        is_calculate_dependent_parameters: bool (calculate the dependent parameters)

    Returns:
        dict (parameter value dictionary)
            key: str (parameter name)
            value: float (parameter value)
    """
    dct = {}
    dct[cn.C_K1] = assignment_function(cn.C_K1)
    dct[cn.C_K2] = assignment_function(cn.C_K2)
    if cn.C_K3 in parameters:
        dct[cn.C_K3] = assignment_function(cn.C_K3)
    if is_calculate_dependent_parameters:
        dct[cn.C_K3] = dct[cn.C_K1] + dct[cn.C_K2]
    dct[cn.C_K4]= assignment_function(cn.C_K4)
    dct[cn.C_K_D]= assignment_function(cn.C_K_D)
    if cn.C_K5 in parameters:
        dct[cn.C_K5] = assignment_function(cn.C_K5)
    if is_calculate_dependent_parameters:
        dct[cn.C_K5] = dct[cn.C_K3] + dct[cn.C_K_D]
    dct[cn.C_K6] = assignment_function(cn.C_K6)
    dct[cn.C_X1_0] = assignment_function(cn.C_X1_0)
    dct[cn.C_X2_0] = assignment_function(cn.C_X2_0)
    dct[cn.C_THETA] = np.sqrt(dct[cn.C_K2]*dct[cn.C_K_D])
    dct[cn.C_T] = assignment_function(cn.C_T)
    return dct
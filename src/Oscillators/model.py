'''Antimony model of the oscillator.'''

import tellurium as te
import numpy as np
import pandas as pd

MODEL = """
J1: S1 -> S2; k1*S1
J2: S2 -> S1; k2*S2
J3: S1 -> 2 S1; k3*S1
J4: S1 -> ; k4
J5: S2 -> ; k5*S1
J6: -> S2; k6

# Parameters are assigned programmatically below
k1 = -1
k2 = -1
k3 = -1
k4 = -1
k5 = -1
k6 = -1
# Initial values assigned here
S1 = 1
S2 = 10

"""
rr = te.loada(MODEL)
period = 1
frequency_in_time = 1/period
frequency_in_radians = frequency_in_time*2*np.pi
PARAM_DCT = {"k1": 1.0, "theta": frequency_in_radians, "k_d": 1, "k4": 5.0, "k6": 5.0}
PARAM_DCT["k2"] = PARAM_DCT["theta"]**2/PARAM_DCT["k_d"]
PARAM_DCT["k3"] = PARAM_DCT["k1"] + PARAM_DCT["k2"]
PARAM_DCT["k5"] = PARAM_DCT["k3"] + PARAM_DCT["k_d"]
PARAM_DCT["x1_0"] = rr["S1"]
PARAM_DCT["x2_0"] = rr["S2"]
ser = pd.Series(PARAM_DCT)
ser = ser.sort_index()
PARAM_DCT = ser.to_dict()
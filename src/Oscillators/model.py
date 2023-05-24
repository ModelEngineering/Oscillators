'''Antimony model of the oscillator.'''
import src.Oscillators.constants as cn

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
k1 = kk1
k2 = kk2
k3 = kk3
k4 = kk4
k5 = kk5
k6 = kk6
# Initial values assigned here
S1 = kS1
S2 = kS2

"""
for  key, value in cn.PARAM_DCT.items():
    stg = 'k%s' % key
    MODEL = MODEL.replace(stg, str(value))
rr = te.loada(MODEL)  # Ensure no syntax errors
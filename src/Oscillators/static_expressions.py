"""
This file contains the static expressions for the Oscillator solution.
"""

# Particular solution for xp_1
xp_1 = "(-k2**2*k4*sp.cos(t*theta) - k2**2*k4 + k2**2*k6*sp.cos(t*theta) + k2**2*k6 - k2*k4*k_d*sp.cos(t*theta) - k2*k4*k_d"
xp_1 += "+k2*k4*theta*sp.sin(t*theta) - k2*k6*theta*sp.sin(t*theta) + k4*k_d*theta*sp.sin(t*theta) + k6*theta**2)/(theta**2*(k2 + k_d))"
# Particular solution for xp_2
xp_2 = "(k2*k4*sp.cos(t*theta) + k2*k4 - k2*k6*sp.cos(t*theta) - k2*k6 + k4*k_d*sp.cos(t*theta) + k4*k_d)/theta**2"
from src.Oscillators import c1, c2, T, phi, r1, r2, t, b1, b2, m1, m2, h1, h2, \
    alpha, omega, theta, S1, S2, k1, k2, k3, k4, k5, k6, k_d, x1, x1_0, x2, x2_0
from src.Oscillators import util

import tellurium as te
import sympy as sp
from sympy import init_printing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import control
import lmfit


def calculateAmplitudePhase(dct):
    """
    Calculates the amplitude and phase.
    """
    a = dct["a"]
    b = dct["b"]
    amplitude = sympy.sqrt(a**2 + b**2)
    phase = sympy.atan(a/b)
    return sympy.simplify(amplitude), sympy.simplify(phase.as_expr())

tt = sympy.Symbol("tt")
uu = sympy.Symbol("uu")
dct = {"a": tt, "b": uu*tt}
amp, phase = calculateAmplitudePhase(dct)
assert("sympy" in str(type(amp)))
assert("atan" in str(type(phase)))
print("OK!")


def makeSinFunction(expression_vec, name_dct=None):
    """
    Transforms the vector expression in terms of cos(theta*t), sin(theta*t) into a function of sin(theta*t + phi)
    
    Args:
        expression_vec: sympy vector Expression
        name_dct: string, values for substituions
    Returns:
        2X1 vector of expressions
    """
    if name_dct is not None:
        symbol_dct = makeSymbolDct(expression_vec, name_dct)
    else:
        symbol_dct = None
    def calculateTimeFunction(expr):
        dct = findSinusoidCoefficients(expr)
        amplitude, phase = calculateAmplitudePhase(dct)
        phase_offset = 0
        if symbol_dct is not None:
            b_subs = dct["b"].subs(symbol_dct)
            if b_subs < 0:
                phase_offset = sympy.pi
        func = amplitude*sympy.sin(theta*t + phase_offset + phase) + dct["c"]
        return func
    #
    x1 = calculateTimeFunction(expression_vec[0])
    x2 = calculateTimeFunction(expression_vec[1])
    return sympy.Matrix([ [x1], [x2] ])

# Tests
soln = makeSinFunction(nonh_solution, name_dct=PARAM_DCT)
vector_func = soln.subs(symbol_dct)
p1 = sympy.plotting.plot(vector_func[0], xlim=[0, 10], line_color="red", show=False)
p2 = sympy.plotting.plot(vector_func[1], xlim=[0, 10], line_color="blue", show=False)
p1.append(p2[0])
p1.show()


print("# In[37]:")


vector_func


print("# In[70]:")


soln = sympy.simplify(makeSinFunction(nonh_solution))
soln


print("# In[72]:")


soln[1]


print("# In[71]:")


print(sympy.python(soln[1]))


# # Designing an Oscillator

# The approach here is to use optimization as a way to search for parameters of the reaction network that produce the desired OCs.

print("# In[63]:")


parameters = lmfit.Parameters()
names = ["k2", "k6", "k4", "k_d", "x20", "x10"]
for name in names:
    parameters.add(name, min=1e-3, max=1e3, value = 1)
parameters


print("# In[ ]:")


def makeResidualCalculator(alpha_1, alpha_2, theta, phi_1, phi_2, omega_1, omega_2, num_point=1000, end_time=5):
    def calculatePhaseOffset(theta):
        if np.abs(theta) > np.pi/2:
            phase_offset = np.pi
        else:
            phase_offset = 0
        return phase_offset
    #
    def calculateReference(alpha, phi, theta, omega):
        times = np.linspace(0, end_time, num_point)
        phase_offset = calculatePhaseOffset(theta)
        return alpha*sin(times*theta + phase_offset + phi) + omega
    #
    x1_ref = calculateReference(alpha_1, theta, phi_1, omega_1)
    x2_ref = calculateReference(alpha_2, theta, phi_2, omega_2)
    def calculateResiduals(params):
        """
        Calculates the results for the parameters.
        """
        for name in names:
            stmt = "%s = params['%x'].value" % (name, name)
            exec(stmt)
        theta = np.sqrt(k2*k_d)
        ####
        # x1
        ####
        numr_omega = -k2**2*k4 + k2**2*k6 - k2*k4*k_d + k6*theta**2
        denom = theta**2*(k2 + k_d)
        omega = numr_omega/denom
        #
        amp_1 = (theta**2*(k2**2*x1_0 + k2**2*x2_0 - k2*k4 + k2*k_d*x1_0 - k4*k_d + theta**2*x2_0)**2 
        amp_2 = (k2**2*k4 - k2**2*k6 + k2*k4*k_d + k2*theta**2*x1_0 - k6*theta**2 + k_d*theta**2*x1_0)**2)
        amp = np.sqrt(sqrt1 + sqrt2)/denom
        numr_theta = k2**2*k4 - k2**2*k6 + k2*k4*k_d + k2*theta**2*x1_0 - k6*theta**2 + k_d*theta**2*x1_0)
        denom_theta = theta*(k2**2*x1_0 + k2**2*x2_0 - k2*k4 + k2*k_d*x1_0 - k4*k_d + theta**2*x2_0)
        phase_offset = calculatePhaseOffset(theta_1)
        theta = np.atan(numr_thera/denom_theta) + phase_offset
        #
        x1 = amp*sin(times*theta + phi) + omega
        ####
        # x2
        ####
        denom = theta**2
        omega = (k2*k4 - k2*k6 + k4*k_d)/denom
        #
        amp_1 = theta**2*(k2*x1_0 + k2*x2_0 - k6 + k_d*x1_0)**2 + (k2*k4 - k2*k6 + k4*k_d - theta**2*x2_0)**2
        amp = np.sqrt(amp_1)/denom
        #
        theta = np.atan((k2*k4 - k2*k6 + k4*k_d - theta**2*x2_0)/(theta*(k2*x1_0 + k2*x2_0 - k6 + k_d*x1_0)))
        phase_offset = calculatePhaseOffset(theta_1)
        theta = theta + phase_offset
        # Calculate the residuals
        residuals_lst = [x for x in x1_ref - x1]
        x2_residuals = [x for x in x2_ref - x2]
        residuals_lst.extend(x2_residulas)
        residuals = np.array(residuals_lst)
        return residuals
    #
    return calculateResiduals


# The parameters of the oscillator are:
# * kinetics parameters: $k_2$, $k_d$
# * initial conditions for species: $x_n (0)$, $n \in \{1, 2\}$
# * forced inputs: $k_4, k_6$
# 
# Given a desired frequency $\theta$, amplitude $h_1, h_2$, and
# midpoint $m_1, m_2$, find values of the parameters that achieve this.

# $r_1 = \frac{k_2}{k_2 + k_d$
# and $r_2 = \frac{\theta}{k_d + k_2}$,
# where $\theta = \sqrt{k_d k_2}$.

# A key fact is that a linear combination of sinusoids at the same frequency is also a sinusoid 
# at the same frequency but with a phase displacement and a new amplitude.
# 
# Let $y = a \times cos(\theta) + b \times sin(\theta)$.
# Let $A = \sqrt{a^2 + b^2}$ and 
# $D = arctan(\frac{b}{a})$.
# Then $y = A \times cos(\theta- D)$.

print("# In[ ]:")


dct = {PARAM_DCT["k2"] + PARAM_DCT["k_d"]: r1, 
       PARAM_DCT["theta"]/ (PARAM_DCT["k_d"] + PARAM_DCT["k2"]): r2}
xf = sympy.simplify(nonh_solution.subs(dct).subs(dct))
#xf = sympy.simplify(nonh_solution.subs({sympy.cos(a*t): -sympy.sin(a*t)}))


print("# In[ ]:")


xf


# Structure the result with sine and cos terms.
# """
# e.as_poly()
# Poly(x*y - 2*x*z + y**2 - 2*y*z, x, y, z, domain='ZZ')
# """
# nonh_solution.as_poly(

print("# In[ ]:")


def removeFactor(sym, factor_str):
    """
    Removes any of the symbolic factors
    """
    result = 1
    for tt in sym.args:
        if not factor_str in str(tt):
            result *= tt
    return result

def separateTerms(sym):
    """
    Separates the terms in a scalar expression into:
        constants
        coefficient of sin(at)
        coefficient of cost(at)
    """
    constants = []
    sints = []
    costs = []
    # Partition the entire expression of a sum of multiplied terms
    for term in sym.expand().args:
        if "cos(" in str(term):
            costs.append(removeFactor(term, "cos"))
        elif "sin(" in str(term):
            sints.append(removeFactor(term, "sin"))
        else:
            constants.append(term)
    return constants, sints, costs

def sumTerms(terms):
    result = 0
    for term in terms:
        result += term
    return sympy.simplify(result)

def calcCoefficients(sym):
    """
    Calculates the coefficients for the constant, sin, and cos terms
    """
    constants, sints, costs = separateTerms(sym)
    return sumTerms(constants), sumTerms(sints), sumTerms(costs)

def getCosine(sym):
    terms = sym.expand().args
    for term in terms:
        if "cos(" in str(term):
            for element in term.args:
                if "cos(" in str(element):
                    return element
            

def refactorSinusoids(sym):
    """
    Combines linear combinations of cos and sin into a single term.
    """
    cos_arg = getCosine(sym).args[0]
    constant, coef_sin, coef_cos = calcCoefficients(sym)
    amplitude = sympy.sqrt(coef_sin**2 + coef_cos**2)
    phase = sympy.atan(coef_sin/coef_cos)
    offset = sympy.Piecewise((0, coef_cos >= 0), (sympy.pi, coef_cos < 0))
    result = constant + amplitude*sympy.cos(cos_arg - phase + offset)
    return result
    

# TESTS
#sym = xf.expand()[0]
#constants, sints, costs = separateTerms(sym)
#const_term, sin_term, cos_term = calcCoefficients(sym)
xf1 = refactorSinusoids(xf.expand()[0])
xf2 = refactorSinusoids(xf.expand()[1])
refactor_vec = sympy.Matrix([xf1, xf2])
            
            
            


# **TO DO**
# 1. Simplify the following.

print("# In[ ]:")


refactor_vec


print("# In[ ]:")


sym = simulateSymbolVector(refactor_vec, SYMBOL_DCT)


print("# In[ ]:")


simulateRR(DCT)


# **TO DO**
# 1. Validate by resimulating
# 1. Approximate for low and high frequency

# Proceed by considering the cases of small $\theta$ and large $\theta$.

print("# In[ ]:")


angles = 2*np.pi*np.array(range(20))*1/20
cost = np.cos(angles)
sint = np.sin(angles)
sin_shift = np.sin(np.pi/2+angles)
plt.scatter(angles, cost, color="red")
plt.scatter(angles, sint, color="green")
plt.plot(angles, sin_shift, color="black")
plt.legend(["cos", "sin", "sin shift"])


print("# In[ ]:")


angles = 2*np.pi*np.array(range(40))*1/20
a1 = 2
a2 = 3
cost = a1*np.cos(angles)
sint = a2*np.sin(angles)
new_a = np.sqrt(a1**2 + a2**2)
new_d = np.arctan(a2/a1)
new_cos = new_a*np.cos(angles - new_d)
plt.scatter(angles, cost+sint)
plt.plot(angles, new_cos)


print("# In[ ]:")


DCT


print("# In[ ]:")


simulateSymbolVector(xf, DCT)


print("# In[ ]:")


simulateRR(DCT)


# ## Solutions

print("# In[ ]:")


xf1 = xf[0]
xf2 = xf[1]


print("# In[ ]:")


xf1


print("# In[ ]:")


xf2


print("# In[ ]:")


eqns = [m1 - (𝑘4*(𝑑+𝑘4)*(𝑢1+𝑢2)), a - sympy.sqrt(d*k4),
       h1 - sympy.Max(a**2, k4*(𝑘4*20 - 𝑢2+ 𝑥1_0*(𝑑+𝑘4)), 
                     a**2/(d*k4*(d + k4)), k4*(𝑘4*𝑥2_0 - 
                                               𝑢2+𝑥1_0*(𝑑+𝑘4))/(d*k4*(d + k4)))]
eqns


print("# In[ ]:")


design_solutions = {a: sympy.sqrt(d*k4),
                    h1: sympy.Max( 
                                  sympy.Abs(((a**2)*x1_0 + x2_0*a**2 + d*u1)/(a*d)),
                                  sympy.Abs(-u1 -u2 + x1_0)/d + x1_0/d),
                    m1: a*u1-a*u2,
                    h2: sympy.Max(
                        sympy.Abs(-a*𝑘4*𝑥2_0 - a*𝑥1_0*(𝑑+𝑘4)+𝑢2*𝑑*𝑘4)/(a**2),
                        
                        sympy.Abs((a*𝑥2_0*d*𝑘4+𝑘4*u2+u1*(d+𝑘4)))/(a**2)
                    ),
                    m2: -k4*u2-u1*(d+k4)/a**2
                                  }


print("# In[ ]:")


design_solutions[a]


# # Notes
# 1. Develop the idea of a UMRA approximation to a non-linear network.
#   1. For mas action networks with two reactants, create two separate uni-reactant reactions. How select the kinetics constants in the approximation to most accurate estimate the original network. The kinetics constants can be approximated using a Taylor's series or by an orthogonal projection for an appropriately defined inner product space.
# 
# 1. Note that the eigenvalues are ${\bf e} = \{ \lambda |  det \left( {\bf A} - \lambda {\bf I} \right) = 0 \}$.
# Since ${\bf A} = {\bf H} + {\bf K}$,
# $ {\bf A} - \lambda {\bf I}  = {\bf H} + {\bf K} - \lambda {\bf I}$.
# 
# 1. Note that $det \left( {\bf A} - \lambda {\bf I} \right)
# = \left( {\bf N V} - \lambda {\bf I} \right)$.
# Further, $det \left( {\bf N} - \lambda {\bf I} \right) \left(
#  {\bf V} - \lambda {\bf I} \right)$
#  $= det \left[ {\bf N}{\bf V} - \lambda {\bf N} - \lambda {\bf V} + I \right]$. Can I use information about the [sum of determinants](https://www.geeksforgeeks.org/eigen-values-and-eigen-vectors/) to figure out
#  how $\lambda$ changes if $k_m$ changes?
# 
# 1. From ["Facts about Eigenvalues"](https://www.adelaide.edu.au/mathslearning/ua/media/120/evalue-magic-tricks-handout.pdf), I know that for a polynomial function $g(x)$, $g({\bf A})$ has eigenvalues $g(\lambda_1), \cdots, g(\lambda_n)$ for the matrix ${\bf A}$ with eigenvalues $\lambda_1, \cdots, \lambda_n$.
# So, it's easy to transform ${\bf A}$ in a way that preserves
# UMRN so that the dominant eigenvalue is at 0 by subtracting along
# the diagonal.
# The challenge is making the this eigenvalue have a non-zero imaginary component.
# 
# 1. Note that in the decomposition into Hermitian and skew Hermitian matrices ${\bf A} = {\bf H} + {\bf K}$, the diagonal of ${\bf K}$ must be zero if ${\bf A}$ is real valued.
# So, how can ${\bf K}$ be transformed to create imaginary eigenvalues?
# 
# 1. Might consider using the polar decomposition, where ${\bf A} = {\bf U} {\bf P}$, where ${\bf U}$ is unitary and ${\bf P}$ is positiv semidefinite. My hypothesis is that if ${\bf U}$ is a rotation other than $n \pi$, then ${\bf A}$ has at least one imaginary eigenvalue. I still have the challenge of making the *dominant* eigenvalue have a non-zero imaginary part.
# 
# 1. Try exploring matrices and their decompositions to understand the criteria for obtaining eigenvalues with a non-zero imaginary part.
# 
# 1. **Issue**: Not getting the correct period
# 

# The homogeneous system is $\dot{\bf x} = {\bf A} {\bf x}$.
# Its solution has the for
# ${\bf x}(t) = \sum _{n=1}^2 c_n {\bf e}_n e^{\lambda_n t}$.

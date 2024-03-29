import numpy as np
import os

# Columns in evaluation data
C_THETA = "theta"
C_ALPHA = "alpha"
C_PHI = "phi"
C_OMEGA = "omega"
C_IS_X1 = "is_x1"
C_FEASIBLEDEV = "feasibledev"
C_ALPHADEV = "alphadev"
C_PHIDEV = "phidev"
C_PREDICTION_ERROR = "prediction_error"  # Fractional SSQ error between ODE model and simulation
C_K1 = "k1"
C_K2 = "k2"
C_KD = "kd"
C_K_D = "k_d"  # Another representation of C_KD
C_K3 = "k3"
C_K4 = "k4"
C_K5 = "k5"
C_K6 = "k6"
C_X1_0 = "x1_0"
C_X2_0 = "x2_0"
C_X1 = "x1"
C_X2 = "x2"
C_T = "t"
C_S1 = "S1"
C_S2 = "S2"
DESIGN_ERROR_LABEL_DCT = {C_FEASIBLEDEV: "feasibility design error",
                          C_ALPHADEV: "amplitude design error",
                          C_PHIDEV: "phase design error",
                          C_PREDICTION_ERROR: "prediction error"}
C_PREDICTION_ERROR = "prediction_error"  # Fractional SSQ error between ODE model and simulation
C_COLUMNS = [C_THETA, C_ALPHA, C_PHI, C_FEASIBLEDEV, C_ALPHADEV, C_PHIDEV, C_K2,
              C_K_D, C_K4, C_K6, C_X1_0, C_X2_0, C_PREDICTION_ERROR]
C_SIMULATION_PARAMETERS = [C_K1, C_K2, C_K3, C_K4, C_K5, C_K6, C_S1, C_S2]
C_MODEL_PARAMETERS = [C_K2, C_K4, C_K_D, C_K6, C_X1_0, C_X2_0]
#
PARAM_DCT = {C_K1: 1, C_K2: 3.913120171941024, C_K3: 4.913120171941024, C_K4: 41.7719311459154, C_K5: 15.001851381229434, 
    C_K6: 92.21558719235689, C_S1: 5.000000000000001, C_S2: 10.491854173592507}
# Add k_d, theta
PARAM_DCT[C_X1_0] = PARAM_DCT[C_S1]
PARAM_DCT[C_X2_0] = PARAM_DCT[C_S2]
PARAM_DCT[C_K_D] = PARAM_DCT[C_K5] - PARAM_DCT[C_K3]
PARAM_DCT[C_THETA] = np.sqrt(PARAM_DCT[C_K2]*PARAM_DCT[C_K_D])
INDEPENDENT_PARAMETERS = [C_K1, C_K2, C_K4, C_K_D, C_K6, C_X1_0, C_X2_0]
DEPENDENT_PARAMETERS = [C_K3, C_K5]
ALL_PARAMETERS = list(INDEPENDENT_PARAMETERS)
ALL_PARAMETERS.extend(DEPENDENT_PARAMETERS)
ALL_PARAMETERS.append(C_T)
OSCILLATION_CHARACTERISTICS = [C_THETA, C_ALPHA, C_PHI, C_OMEGA]

K1_VALUE =1  # Value used for k1

# Paths
PROJECT_DIR = os.path.dirname(__file__)
for _ in range(2):
    PROJECT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PLOT_DIR = os.path.join(PROJECT_DIR, "plots")
EVALUATION_CSV = os.path.join(DATA_DIR, "evaluation_data.csv")
EVALUATION_PLOT_PATH = os.path.join(PLOT_DIR, "evaluation_plot.pdf")
HISTOGRAM_PLOT_PATH = os.path.join(PLOT_DIR, "histogram_plot.pdf")

# Metrics
C_ALPHA1 = "alpha1"
C_ALPHA2 = "alpha2"
C_PHI1 = "phi1"
C_PHI2 = "phi2"
C_OMEGA1 = "omega1"
C_OMEGA2 = "omega2"
C_NONOSCILLATING = "nonoscillating"
C_INFEASIBLE = "infeasible"
C_SAMPLE_SIZE = "sample_size"
METRICS = [C_ALPHA1, C_ALPHA2, C_PHI1, C_PHI2, C_OMEGA1, C_OMEGA2, C_NONOSCILLATING, C_INFEASIBLE, C_SAMPLE_SIZE, C_THETA]
C_MEAN = "mean"
C_STD = "std"
C_OTHER = "other"
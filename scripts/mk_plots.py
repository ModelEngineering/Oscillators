"""Creates Plots for Paper"""

from src.Oscillators.evaluator import Evaluator
from src.Oscillators.solver import Solver
import src.Oscillators.constants as cn
from src.Oscillators.sensitivity_analyzer import SensitivityAnalyzer

import matplotlib.pyplot as plt
import os
import shutil

SPECIES = ["x1", "x2", "both"]
METRICS = ["feasibledev", "alphadev", "phidev", "prediction_error"]
#
CSV_PATH_PAT = os.path.join(cn.DATA_DIR, "k1_is_1", "evaluation_data_%s.csv")
CSV_PATHS = [CSV_PATH_PAT % s for s in SPECIES]
EVALUATION_PLOT_PATH_PAT = os.path.join(cn.PLOT_DIR, "evaluation_plot_%s_%s.pdf")   #  metric, species
HISTOGRAM_PLOT_PATH_PAT = os.path.join(cn.PLOT_DIR, "histogram_plot_%s.pdf")   #  species
EVALUATE_MODEL_PLOT = os.path.join(cn.PLOT_DIR, "evaluate_model.pdf")
EVALUATION_PLOT_PATH_DCT = {}
for metric in METRICS:
    for species in SPECIES:
        EVALUATION_PLOT_PATH_DCT[(metric, species)] = EVALUATION_PLOT_PATH_PAT % (metric, species)
HISTOGRAM_PLOT_PATH_DCT = {s: HISTOGRAM_PLOT_PATH_PAT % s for s in SPECIES}


# Plots
for species in SPECIES:
    csv_path = CSV_PATH_PAT % species
    print("\n\nPlotting %s" % csv_path)
    Evaluator.plotParameterHistograms(output_path=HISTOGRAM_PLOT_PATH_DCT[species], csv_path=csv_path)
    for metric in METRICS:
        Evaluator.plotDesignErrors(metric, csv_path=csv_path,
                                  plot_path=EVALUATION_PLOT_PATH_DCT[(metric, species)], vmin=-1, vmax=1)

# Plot accuracy of predictions
solver = Solver()
solver.solve(is_check=False)
solver.plotManyFits(is_plot=True, output_path=EVALUATE_MODEL_PLOT)

# Create paper figure files
def copyFile(src, figure_num):
    dst = os.path.join(cn.PLOT_DIR, "Figure_%d.pdf" % figure_num)
    shutil.copyfile(src, dst)
#
copyFile(EVALUATE_MODEL_PLOT, 3)
copyFile(EVALUATION_PLOT_PATH_DCT[("alphadev", "x1")], 4)
copyFile(EVALUATION_PLOT_PATH_DCT[("alphadev", "both")], 5)
copyFile(EVALUATION_PLOT_PATH_DCT[("phidev", "both")], 6)
copyFile(HISTOGRAM_PLOT_PATH_PAT % "both", 7)

# Sensitivity Analysis
analyzer = SensitivityAnalyzer()
analyzer.plotMetrics()
plot_path = os.path.join(cn.PLOT_DIR, "sensitivity_analysis.pdf")
plt.savefig(os.path.join(plot_path))
copyFile(plot_path, 8)
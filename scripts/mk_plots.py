"""Creates Plots for Paper"""

from src.Oscillators.evaluator import Evaluator
import src.Oscillators.constants as cn

import os

SPECIES = ["x1", "x2", "both"]
METRICS = ["feasibledev", "alphadev", "phidev", "prediction_error"]
#
CSV_PATH_PAT = os.path.join(cn.DATA_DIR, "evaluation_data_%s.csv")
CSV_PATHS = [CSV_PATH_PAT % s for s in SPECIES]
EVALUATION_PLOT_PATH_PAT = os.path.join(cn.PLOT_DIR, "evaluation_plot_%s_%s.pdf")   #  metric, species
HISTOGRAM_PLOT_PATH_PAT = os.path.join(cn.PLOT_DIR, "histogram_plot_%s.pdf")   #  species
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
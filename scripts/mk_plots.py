"""Creates Plots for Paper"""

from src.Oscillators.evaluator import Evaluator
import src.Oscillators.constants as cn

import os

CSV_PATH_PAT = os.path.join(cn.DATA_DIR, "evaluation_data_%s.csv")
CSV_PATH_X1 = CSV_PATH_PAT % "x1"
CSV_PATH_X2 = CSV_PATH_PAT % "x2"
CSV_PATH_BOTH = CSV_PATH_PAT % "both"
CSV_PATHS = [CSV_PATH_X1, CSV_PATH_X2, CSV_PATH_BOTH]
EVALUTIONA_PLOT_PATH_PAT = os.path.join(cn.PLOT_DIR, "evaluation_plot_%s_%s.pdf")   #  metric, csv_path
METRICS = ["feasibledev", "alphadev", "phidev", "prediction_error"]
EVALUATION_PLOT_PATH_DCT = {(m, p): EVALUTIONA_PLOT_PATH_PAT % (m, p) for m, p in zip(METRICS, CSV_PATHS)}:while
HISTOGRAM_PLOT_PATH_DCT = {p: EVALUTIONA_PLOT_PATH_PAT % p for p in CSV_PATHS}:while


# Plots
for csv_path in CSV_PATHS:
    print("\n\nPlotting %s" % csv_path)
    Evaluator.plotParameterHistograms(output_path=HISTOGRAM_PLOT_PATH_DCT[csv_path], csv_path=csv_path)
    for metric in METRICS:
        Evaluator.plotDesignErrors(metric, csv_path=csv_path,
                                  plot_path=EVALUATION_PLOT_PATH_DCT[(metric, csv_path)], vmin=-1, vmax=1)
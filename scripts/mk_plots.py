"""Creates Plots for Paper"""

from src.Oscillators.evaluator import Evaluator

Evaluator.plotEvaluationData("feasibledev", plot_path="plotEvaluationData_feas.pdf", vmin=-1, vmax=1)
Evaluator.plotEvaluationData("alphadev", plot_path="plotEvaluationData_alpha.pdf", vmin=-1, vmax=1)
Evaluator.plotEvaluationData("phidev", plot_path="plotEvaluationData_phi.pdf", vmin=-1, vmax=1)
Evaluator.plotEvaluationData("prediction_error", plot_path="plotEvaluationData_prediction_error.pdf", vmin=-1, vmax=1)
Evaluator.plotParameterHistograms(output_path="plotParameterHistograms.pdf")

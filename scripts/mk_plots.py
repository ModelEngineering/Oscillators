"""Creates Plots for Paper"""

from src.Oscillators.evaluator import Evaluator

Evaluator.plotDesignErrors("feasibledev", plot_path="plotEvaluationData_feas.pdf", vmin=-1, vmax=1)
Evaluator.plotDesignErrors("alphadev", plot_path="plotEvaluationData_alpha.pdf", vmin=-1, vmax=1)
Evaluator.plotDesignErrors("phidev", plot_path="plotEvaluationData_phi.pdf", vmin=-1, vmax=1)
Evaluator.plotDesignErrors("prediction_error", plot_path="plotEvaluationData_prediction_error.pdf", vmin=-1, vmax=1)
Evaluator.plotParameterHistograms(output_path="plotParameterHistograms.pdf")

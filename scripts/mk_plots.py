"""Creates Plots for Paper"""

from src.Oscillators.designer import Designer

Designer.plotEvaluationData("feasibledev", plot_path="plotEvaluationData_feas.pdf", vmin=-1, vmax=1)
Designer.plotEvaluationData("alphadev", plot_path="plotEvaluationData_alpha.pdf", vmin=-1, vmax=1)
Designer.plotEvaluationData("phidev", plot_path="plotEvaluationData_phi.pdf", vmin=-1, vmax=1)
Designer.plotParameterHistograms(output_path="plotParameterHistograms.pdf")

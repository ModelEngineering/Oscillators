"""Creates the data used in the paper"""

from src.Oscillators.evaluator import Evaluator

import os

CSV_PATH_PAT = os.path.join(os.path.dirname(__file__), "evaluation_data_%s.csv")

Evaluator.makeData(is_x1=True, csv_path=CSV_PATH_PAT % "x1")
Evaluator.makeData(is_x1=False, csv_path=CSV_PATH_PAT % "x2")

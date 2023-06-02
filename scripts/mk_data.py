"""Creates the data used in the paper"""

import src.Oscillators.constants as cn
from src.Oscillators.evaluator import Evaluator

import os

CSV_PATH_PAT = os.path.join(cn.DATA_DIR, "evaluation_data_%s.csv")
CSV_PATH_X1 = CSV_PATH_PAT % "x1"
CSV_PATH_X2 = CSV_PATH_PAT % "x2"
CSV_PATH_BOTH = CSV_PATH_PAT % "both"
CSV_PATHS = [CSV_PATH_X1, CSV_PATH_X2, CSV_PATH_BOTH]
#

print("****Creating data for x1****")
Evaluator.makeData(is_x1=True, is_both=False, csv_path=CSV_PATH_X1)
print("\n\n****Creating data for x2****")
Evaluator.makeData(is_x1=False, is_both=False, csv_path=CSV_PATH_X2)
print("\n\n****Creating data for both****")
Evaluator.makeData(is_both=True, csv_path=CSV_PATH_BOTH)
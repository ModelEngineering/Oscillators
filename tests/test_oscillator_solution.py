import unittest
from src.Oscillators.oscillator_solution import OscillatorSolution
from src.Oscillators.util import TIMES, MODEL

IGNORE_TEST = False
IS_PLOT = False

class TestOscillatorSolution(unittest.TestCase):

    def setUp(self):
        self.soln = OscillatorSolution()

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.soln.A_mat.shape, (2, 2))

    def testGetSolution(self):
        if IGNORE_TEST:
            return
        self.soln.getSolution(is_check=True)
        self.assertEqual(self.soln.x_vec.shape, (2, 1))

if __name__ == "__main__":
    unittest.main()
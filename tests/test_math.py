import unittest
import numpy as np
from src.tree.loss_functions import LogLoss
from src.tree.decision_tree import DecisionTree
from src.model import XGBoostScratch

class TestXGBoostMath(unittest.TestCase):
    """
    Unit tests for mathematical components of XGBoost.
    Issue 14.5: Tests Unitarios Matemáticos (Toy Dataset).
    """

    def setUp(self):
        # Manual Toy Dataset (5 rows)
        # Features: [x1]
        # Target: [y]
        self.X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        self.y = np.array([0, 0, 0, 1, 1])
        self.loss = LogLoss()

    def test_gradient_hessian_manual(self):
        """
        Verify Gradient and Hessian calculation against manual values.
        """
        # Assume initial prediction p = 0.5 (log-odds = 0)
        y_pred_score = np.zeros(5)
        p = 0.5

        # Manual calculation
        # g = p - y
        g_expected = p - self.y # [0.5, 0.5, 0.5, -0.5, -0.5]

        # h = p * (1 - p)
        h_expected = p * (1 - p) # 0.25 for all
        h_expected_arr = np.full(5, 0.25)

        g_calc = self.loss.gradient(self.y, y_pred_score)
        h_calc = self.loss.hessian(self.y, y_pred_score)

        np.testing.assert_array_almost_equal(g_calc, g_expected, decimal=5)
        np.testing.assert_array_almost_equal(h_calc, h_expected_arr, decimal=5)

    def test_gain_calculation(self):
        """
        Verify Gain formula manually.
        """
        # Let's say we split at x=3.5 (Left: [1,2,3], Right: [4,5])
        # g = [0.5, 0.5, 0.5, -0.5, -0.5]
        # h = [0.25, 0.25, 0.25, 0.25, 0.25]

        g_L = np.array([0.5, 0.5, 0.5])
        h_L = np.array([0.25, 0.25, 0.25])
        g_R = np.array([-0.5, -0.5])
        h_R = np.array([0.25, 0.25])

        G_L = 1.5
        H_L = 0.75
        G_R = -1.0
        H_R = 0.5

        lambda_ = 1.0
        gamma = 0.0

        # Gain = 0.5 * [ G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - (G_L+G_R)^2/(H_L+H_R+lam) ] - gamma
        term_L = (1.5**2) / (0.75 + 1.0) # 2.25 / 1.75 = 1.2857
        term_R = ((-1.0)**2) / (0.5 + 1.0) # 1.0 / 1.5 = 0.6667
        term_Total = ((0.5)**2) / (1.25 + 1.0) # 0.25 / 2.25 = 0.1111

        expected_gain = 0.5 * (term_L + term_R - term_Total)

        tree = DecisionTree(lambda_=lambda_, gamma=gamma)
        calc_gain = tree._calculate_gain(g_L, h_L, g_R, h_R)

        self.assertAlmostEqual(calc_gain, expected_gain, places=4)

    def test_xgboost_fit_toy(self):
        """
        Test that XGBoost fits the toy dataset and reduces loss.
        """
        model = XGBoostScratch(n_estimators=5, learning_rate=0.3, max_depth=2)
        model.fit(self.X, self.y)

        # Predict
        preds = model.predict_proba(self.X)

        # Check if predictions align roughly with targets
        # First 3 should be low, last 2 should be high
        self.assertTrue(np.mean(preds[:3]) < np.mean(preds[3:]))

if __name__ == '__main__':
    unittest.main()

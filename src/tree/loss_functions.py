import numpy as np

class LogLoss:
    """
    LogLoss (Binary Cross Entropy) for Binary Classification.
    """
    def __init__(self):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, y_true, y_pred_score):
        """
        First order derivative (gradient) with respect to the predicted score (logits).
        g_i = p_i - y_i
        """
        p = self.sigmoid(y_pred_score)
        return p - y_true

    def hessian(self, y_true, y_pred_score):
        """
        Second order derivative (hessian) with respect to the predicted score (logits).
        h_i = p_i * (1 - p_i)
        """
        p = self.sigmoid(y_pred_score)
        return p * (1 - p)

    def loss(self, y_true, y_pred_score):
        p = self.sigmoid(y_pred_score)
        # Clip to prevent log(0)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

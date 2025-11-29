import numpy as np

class LogLoss:
    """
    LogLoss (Binary Cross Entropy) for Binary Classification.
    Supports class weighting via scale_pos_weight.
    """
    def __init__(self, scale_pos_weight=1.0):
        self.scale_pos_weight = scale_pos_weight

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, y_true, y_pred_score):
        """
        First order derivative (gradient) with respect to the predicted score (logits).
        g_i = p_i - y_i

        With scale_pos_weight (w) for positive class:
        If y=1: g = (p - 1) * w = (p - y) * w
        If y=0: g = p - 0 = p - y
        """
        p = self.sigmoid(y_pred_score)
        grad = p - y_true

        # Apply weight only where y_true == 1
        # Vectorized operation
        if self.scale_pos_weight != 1.0:
            # We want to weight the gradient for the positive class.
            # Original Gradient Boosting logic with weights:
            # Loss = - [w * y * log(p) + (1-y) * log(1-p)]
            # dLoss/d(log(odds)) = p - y  <-- standard
            # Weighted:
            # If y=1: term is -w * log(p). deriv w.r.t logits (z): -w * (1 - p) = w * (p - 1).
            # If y=0: term is -log(1-p). deriv w.r.t logits (z): p.
            # So:
            # If y=1: g = w * (p - 1)
            # If y=0: g = p
            # Can be written as: g = p * (y*w + (1-y)) - y*w
            # Wait, let's verify.
            # standard g = p - y.
            # if y=1: p - 1. weighted: w(p-1).
            # if y=0: p - 0. weighted: p.

            # Implementation:
            weights = np.where(y_true == 1, self.scale_pos_weight, 1.0)
            grad = grad * weights

        return grad

    def hessian(self, y_true, y_pred_score):
        """
        Second order derivative (hessian) with respect to the predicted score (logits).
        h_i = p_i * (1 - p_i)

        With weights:
        h = p * (1 - p) * weight
        """
        p = self.sigmoid(y_pred_score)
        hess = p * (1 - p)

        if self.scale_pos_weight != 1.0:
            weights = np.where(y_true == 1, self.scale_pos_weight, 1.0)
            hess = hess * weights

        return hess

    def loss(self, y_true, y_pred_score):
        p = self.sigmoid(y_pred_score)
        # Clip to prevent log(0)
        p = np.clip(p, 1e-15, 1 - 1e-15)

        if self.scale_pos_weight != 1.0:
            weights = np.where(y_true == 1, self.scale_pos_weight, 1.0)
            return -np.mean(weights * (y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
        else:
            return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

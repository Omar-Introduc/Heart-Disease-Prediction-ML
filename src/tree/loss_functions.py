import numpy as np

class Loss:
    """
    Base class for loss functions.
    """
    def gradient(self, y_true, y_pred):
        """
        Calculates the gradient of the loss function.
        """
        raise NotImplementedError()

    def hessian(self, y_true, y_pred):
        """
        Calculates the Hessian of the loss function.
        """
        raise NotImplementedError()

class SquaredError(Loss):
    """
    Squared Error loss function for regression.
    """
    def gradient(self, y_true, y_pred):
        return 2 * (y_pred - y_true)

    def hessian(self, y_true, y_pred):
        return np.full_like(y_true, 2)

class LogLoss(Loss):
    """
    Logistic Loss function for binary classification.
    """
    def gradient(self, y_true, y_pred):
        return y_pred - y_true

    def hessian(self, y_true, y_pred):
        return y_pred * (1 - y_pred)

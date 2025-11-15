import numpy as np
from src.tree.decision_tree import DecisionTree
from src.tree.loss_functions import LogLoss
import src.config as config

class XGBoost:
    """
    XGBoost model implemented from scratch.
    """
    def __init__(self, n_estimators=config.N_ESTIMATORS,
                 learning_rate=config.LEARNING_RATE,
                 max_depth=config.MAX_DEPTH,
                 min_samples_split=config.MIN_SAMPLES_SPLIT,
                 reg_lambda=config.REG_LAMBDA,
                 gamma=config.GAMMA):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.loss = LogLoss()
        self.trees = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=None):
        """
        Trains the XGBoost model.
        """
        n_samples, n_features = X.shape
        self.base_pred = 0.0
        y_pred = np.full((n_samples, 1), self.base_pred)
        best_loss = np.inf
        epochs_without_improvement = 0

        for i in range(self.n_estimators):
            grad = self.loss.gradient(y, self._sigmoid(y_pred))
            hess = self.loss.hessian(y, self._sigmoid(y_pred))

            tree = DecisionTree(
                loss=self.loss,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma
            )
            tree.fit(X, grad, hess)

            update = tree.predict(X).reshape(-1, 1)
            y_pred += self.learning_rate * update

            self.trees.append(tree)

            if X_val is not None and y_val is not None and early_stopping_rounds is not None:
                val_pred = self.predict(X_val)
                val_loss = self.loss.loss(y_val, val_pred)
                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= early_stopping_rounds:
                        print(f"Early stopping at iteration {i}")
                        break

    def predict(self, X):
        """
        Makes predictions with the trained XGBoost model.
        """
        y_pred = np.full((X.shape[0], 1), self.base_pred)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X).reshape(-1, 1)

        return self._sigmoid(y_pred)

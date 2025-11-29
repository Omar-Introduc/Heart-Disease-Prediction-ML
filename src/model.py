import numpy as np
from src.tree.decision_tree import DecisionTree
from src.tree.loss_functions import LogLoss
from src.interfaces import HeartDiseaseModel


class XGBoostScratch(HeartDiseaseModel):
    """
    XGBoost implementation from scratch.
    Implements the HeartDiseaseModel protocol.
    """

    def __init__(
        self, n_estimators=10, learning_rate=0.1, max_depth=3, lambda_=1.0, gamma=0.0
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda_ = lambda_
        self.gamma = gamma
        self.trees = []
        self.loss_func = LogLoss()
        self.base_score = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Initial prediction (log-odds)
        # Assuming base_score = 0.5 -> log(p/(1-p)) = 0
        self.base_score = 0.5
        y_pred = np.full(y.shape, 0.0)  # Log-odds for 0.5 is 0.0

        for i in range(self.n_estimators):
            # Calculate Gradients and Hessians
            g = self.loss_func.gradient(y, y_pred)
            h = self.loss_func.hessian(y, y_pred)

            # Fit a new tree to the gradients
            tree = DecisionTree(
                max_depth=self.max_depth, lambda_=self.lambda_, gamma=self.gamma
            )
            tree.fit(X, g, h)
            self.trees.append(tree)

            # Update predictions
            # y_new = y_old + eta * f_t(x)
            update = tree.predict(X)
            y_pred += self.learning_rate * update

            # Optional: Print loss
            # current_loss = self.loss_func.loss(y, y_pred)
            # print(f"Iter {i}: Loss = {current_loss:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.full(X.shape[0], 0.0)  # Start with initial log-odds 0
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        return self.loss_func.sigmoid(y_pred)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

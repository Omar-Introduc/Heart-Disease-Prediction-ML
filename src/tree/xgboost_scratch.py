import numpy as np
from src.tree.decision_tree import DecisionTree
from src.tree.loss_functions import LogLoss
from src.interfaces import HeartDiseaseModel

class XGBoostScratch(HeartDiseaseModel):
    """
    XGBoost (Extreme Gradient Boosting) implementation from scratch.

    This class implements the sequential boosting logic, where each new tree
    attempts to correct the errors (gradients) of the previous ensemble.

    It adheres to the HeartDiseaseModel protocol for compatibility with the project's ecosystem.
    """

    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, lambda_=1.0, gamma=0.0):
        """
        Initialize the XGBoost model.

        Args:
            n_estimators (int): Number of boosting rounds (trees).
            learning_rate (float): Step size shrinkage used in update to prevent overfitting.
            max_depth (int): Maximum depth of a tree.
            lambda_ (float): L2 regularization term on weights.
            gamma (float): Minimum loss reduction required to make a further partition on a leaf node.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda_ = lambda_
        self.gamma = gamma
        self.trees = []
        self.loss_func = LogLoss()
        self.base_score = 0.5 # Initial probability prediction

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the XGBoost model on the given dataset.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector (0 or 1).
        """
        # Initialize predictions with the base score (log-odds 0.0 for p=0.5)
        self.base_score = 0.5
        # Initial log(odds) prediction. log(0.5 / (1-0.5)) = log(1) = 0.0
        y_pred = np.full(y.shape, 0.0)

        # Clear any existing trees
        self.trees = []

        # Sequential Boosting Loop
        for i in range(self.n_estimators):
            # 1. Calculate Gradients (1st order derivative) and Hessians (2nd order derivative)
            #    of the loss function with respect to current predictions.
            #    Gradient tells us the direction of the error.
            #    Hessian tells us the curvature (confidence) of the error.
            g = self.loss_func.gradient(y, y_pred)
            h = self.loss_func.hessian(y, y_pred)

            # 2. Fit a new Decision Tree (Weak Learner) to the gradients/hessians.
            #    The tree tries to predict the structure of the gradients.
            tree = DecisionTree(
                max_depth=self.max_depth,
                lambda_=self.lambda_,
                gamma=self.gamma
            )
            tree.fit(X, g, h)
            self.trees.append(tree)

            # 3. Update the ensemble's predictions.
            #    y_new = y_old + learning_rate * tree_output
            #    The tree output is the optimal weight w* for each leaf.
            update = tree.predict(X)
            y_pred += self.learning_rate * update

            # Optional: Monitor loss convergence
            # current_loss = self.loss_func.loss(y, y_pred)
            # print(f"Tree {i+1}/{self.n_estimators} - Loss: {current_loss:.6f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of the positive class (Heart Disease).

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Probabilities [0, 1].
        """
        # Start with the initial base prediction (log-odds 0.0)
        y_pred = np.full(X.shape[0], 0.0)

        # Add the contribution of each tree in the ensemble
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        # Apply Sigmoid function to convert log-odds to probability
        return self.loss_func.sigmoid(y_pred)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary class labels.

        Args:
            X (np.ndarray): Feature matrix.
            threshold (float): Decision threshold.

        Returns:
            np.ndarray: Binary labels (0 or 1).
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

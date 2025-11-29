import numpy as np
import copy
from src.tree.decision_tree import DecisionTree
from src.tree.loss_functions import LogLoss

class XGBoostScratch:
    """
    XGBoost implementation from scratch using Gradient Boosting on Decision Trees.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, lambda_=1.0, gamma=0.0, scale_pos_weight=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda_ = lambda_
        self.gamma = gamma
        self.scale_pos_weight = scale_pos_weight
        self.trees = []
        self.loss_func = LogLoss(scale_pos_weight=self.scale_pos_weight)
        self.base_pred = 0.0

    def fit(self, X, y):
        """
        Fits the XGBoost model to the data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector (binary 0/1).
        """
        # Ensure input is numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # Initial prediction (log(odds) = 0.0 implies p=0.5)
        # If classes are imbalanced, starting with log(pos/neg) is better, but 0.0 is standard for educational scratch
        self.base_pred = 0.0
        y_pred = np.full(y.shape, self.base_pred)

        self.trees = []

        for _ in range(self.n_estimators):
            # Calculate gradients and hessians
            g = self.loss_func.gradient(y, y_pred)
            h = self.loss_func.hessian(y, y_pred)

            # Fit a tree to the gradients
            # Note: DecisionTree in this repo expects (X, g, h) and optimizes split based on Gain(g, h)
            tree = DecisionTree(
                max_depth=self.max_depth,
                lambda_=self.lambda_,
                gamma=self.gamma
            )
            tree.fit(X, g, h)
            self.trees.append(tree)

            # Update predictions
            # The tree output (leaf weights) is already the optimal step step_size * direction
            # We scale it by learning_rate
            update = tree.predict(X)
            y_pred += self.learning_rate * update

    def predict_proba(self, X):
        """
        Predicts class probabilities for X.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Probabilities of class 1.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        y_pred = np.full(X.shape[0], self.base_pred)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        return self.loss_func.sigmoid(y_pred)

    def predict(self, X, threshold=0.5):
        """
        Predicts binary class labels for X.

        Args:
            X (np.ndarray): Feature matrix.
            threshold (float): Threshold for classification.

        Returns:
            np.ndarray: Predicted binary labels.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

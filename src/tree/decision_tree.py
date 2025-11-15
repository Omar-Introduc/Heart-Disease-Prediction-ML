import numpy as np
from src.tree.node import Node

class DecisionTree:
    """
    Represents a decision tree for gradient boosting.
    """
    def __init__(self, loss, min_samples_split=2, max_depth=100, n_features=None, 
                 learning_rate=0.1, reg_lambda=1.0, gamma=0.0):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.loss = loss
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.root = None

    def fit(self, X, grad, hess):
        """
        Builds the decision tree using gradient and hessian.
        """
        self.n_features = X.shape[1] if self.n_features is None else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, grad, hess)

    def _grow_tree(self, X, grad, hess, depth=0):
        n_samples, n_feats = X.shape
        
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_value = self._calculate_leaf_value(grad, hess)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_thresh = self._best_split(X, grad, hess, feat_idxs)

        if best_feature is None:
            leaf_value = self._calculate_leaf_value(grad, hess)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], grad[left_idxs], hess[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], grad[right_idxs], hess[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, grad, hess, feat_idxs):
        best_gain = -np.inf
        split_idx, split_thresh = None, None
        
        sum_grad = np.sum(grad)
        sum_hess = np.sum(hess)

        for feat_idx in feat_idxs:
            X_col = X[:, feat_idx]
            thresholds = np.unique(X_col)

            for thr in thresholds:
                left_idxs, right_idxs = self._split(X_col, thr)
                
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue
                
                grad_left, grad_right = grad[left_idxs], grad[right_idxs]
                hess_left, hess_right = hess[left_idxs], hess[right_idxs]
                
                sum_grad_left, sum_grad_right = np.sum(grad_left), np.sum(grad_right)
                sum_hess_left, sum_hess_right = np.sum(hess_left), np.sum(hess_right)
                
                gain = 0.5 * (
                    (sum_grad_left**2 / (sum_hess_left + self.reg_lambda)) +
                    (sum_grad_right**2 / (sum_hess_right + self.reg_lambda)) -
                    (sum_grad**2 / (sum_hess + self.reg_lambda))
                ) - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr
        
        return split_idx, split_thresh

    def _calculate_leaf_value(self, grad, hess):
        return -np.sum(grad) / (np.sum(hess) + self.reg_lambda) * self.learning_rate

    def _split(self, X_col, split_thresh):
        left_idxs = np.argwhere(X_col <= split_thresh).flatten()
        right_idxs = np.argwhere(X_col > split_thresh).flatten()
        return left_idxs, right_idxs

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

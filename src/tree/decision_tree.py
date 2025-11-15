import numpy as np
from collections import Counter
from src.tree.node import Node

class DecisionTree:
    """
    Represents a decision tree classifier.

    Attributes:
        min_samples_split (int): Minimum number of samples required to split a node.
        max_depth (int): Maximum depth of the tree.
        n_features (int): Number of features to consider for splitting.
        root (Node): The root node of the tree.
    """
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None, criterion='entropy'):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        """
        Builds the decision tree.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).
        """
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_col = X[:, feat_idx]
            thresholds = np.unique(X_col)

            for thr in thresholds:
                gain = self._information_gain(y, X_col, thr, self.criterion)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _information_gain(self, y, X_col, threshold, criterion):
        if criterion == 'entropy':
            parent_impurity = self._entropy(y)
            impurity_func = self._entropy
        elif criterion == 'gini':
            parent_impurity = self._gini(y)
            impurity_func = self._gini
        else:
            raise ValueError("Invalid criterion. Choose 'entropy' or 'gini'.")

        # Create children
        left_idxs, right_idxs = self._split(X_col, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Calculate the weighted average impurity of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        i_l, i_r = impurity_func(y[left_idxs]), impurity_func(y[right_idxs])
        child_impurity = (n_l / n) * i_l + (n_r / n) * i_r

        # Calculate IG
        information_gain = parent_impurity - child_impurity
        return information_gain

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _gini(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1 - np.sum([p**2 for p in ps])

    def _split(self, X_col, split_thresh):
        left_idxs = np.argwhere(X_col <= split_thresh).flatten()
        right_idxs = np.argwhere(X_col > split_thresh).flatten()
        return left_idxs, right_idxs

    def predict(self, X):
        """
        Makes predictions for a given dataset.

        Args:
            X (np.ndarray): Data to make predictions on.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        """
        Finds the most common label in a set of labels.

        Args:
            y (np.ndarray): A numpy array of labels.

        Returns:
            The most common label.
        """
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

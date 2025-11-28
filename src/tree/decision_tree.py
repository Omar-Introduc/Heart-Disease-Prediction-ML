import numpy as np

class DecisionTree:
    """
    A single Decision Tree for XGBoost.
    Uses gradients and hessians to calculate split gain and leaf weights.
    """
    def __init__(self, max_depth=3, min_samples_split=2, lambda_=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lambda_ = lambda_  # L2 regularization term
        self.gamma = gamma      # Minimum gain required to split
        self.tree = None

    def fit(self, X, g, h):
        """
        Fits the tree using features X, gradients g, and hessians h.
        """
        # Ensure X is a numpy array for consistent slicing
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        self.tree = self._build_tree(X, g, h, depth=0)

    def _build_tree(self, X, g, h, depth):
        # Calculate weight for current node
        weight = self._calculate_leaf_weight(g, h)

        # Stop conditions
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            return {'leaf': True, 'weight': weight}

        # Find best split
        best_split = self._find_best_split(X, g, h)

        if best_split is None or best_split['gain'] < 0: # Should be gain <= gamma? The formula says Gain - gamma. If result is negative, don't split.
             return {'leaf': True, 'weight': weight}

        # Recursively build left and right subtrees
        left_tree = self._build_tree(best_split['X_left'], best_split['g_left'], best_split['h_left'], depth + 1)
        right_tree = self._build_tree(best_split['X_right'], best_split['g_right'], best_split['h_right'], depth + 1)

        return {
            'leaf': False,
            'feature_idx': best_split['feature_idx'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree,
            'weight': weight # Store weight just in case
        }

    def _calculate_leaf_weight(self, g, h):
        """
        Calculates the optimal weight for a leaf: w* = - Sum(g) / (Sum(h) + lambda)
        """
        G = np.sum(g)
        H = np.sum(h)
        return -G / (H + self.lambda_)

    def _calculate_gain(self, g_L, h_L, g_R, h_R):
        """
        Calculates the gain of a split:
        Gain = 0.5 * [ (G_L^2 / (H_L + lambda)) + (G_R^2 / (H_R + lambda)) - ((G_L + G_R)^2 / (H_L + H_R + lambda)) ] - gamma
        """
        G_L = np.sum(g_L)
        H_L = np.sum(h_L)
        G_R = np.sum(g_R)
        H_R = np.sum(h_R)

        term_L = (G_L**2) / (H_L + self.lambda_)
        term_R = (G_R**2) / (H_R + self.lambda_)
        term_Total = ((G_L + G_R)**2) / (H_L + H_R + self.lambda_)

        gain = 0.5 * (term_L + term_R - term_Total) - self.gamma
        return gain

    def _find_best_split(self, X, g, h):
        best_gain = -float('inf')
        best_split = None
        n_features = X.shape[1]

        for feature_idx in range(n_features):
            # Sort data by feature
            # Optimization: Pre-sort or use histograms (approx) but for this "from scratch" version, simple sorting is fine.
            # However, checking every unique value as threshold is slow.
            # For simplicity in this sprint, we can iterate unique values.

            thresholds = np.unique(X[:, feature_idx])

            # Optimization: Vectorized scan is hard in pure python loop structure,
            # but we can iterate thresholds.

            for threshold in thresholds:
                # Split
                mask_left = X[:, feature_idx] <= threshold
                mask_right = ~mask_left

                if np.sum(mask_left) == 0 or np.sum(mask_right) == 0:
                    continue

                g_L, h_L = g[mask_left], h[mask_left]
                g_R, h_R = g[mask_right], h[mask_right]

                gain = self._calculate_gain(g_L, h_L, g_R, h_R)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'gain': gain,
                        'X_left': X[mask_left],
                        'X_right': X[mask_right],
                        'g_left': g_L,
                        'g_right': g_R,
                        'h_left': h_L,
                        'h_right': h_R
                    }

        return best_split

    def predict(self, X):
        if not isinstance(X, np.ndarray):
             X = np.array(X)
        predictions = np.array([self._predict_single(x, self.tree) for x in X])
        return predictions

    def _predict_single(self, x, tree):
        if tree['leaf']:
            return tree['weight']

        if x[tree['feature_idx']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])

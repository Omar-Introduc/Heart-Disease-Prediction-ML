import numpy as np

class Node:
    """
    Represents a node in the decision tree.

    Attributes:
        feature_index (int): Index of the feature used for splitting at this node.
        threshold (float): Threshold value for the feature used for splitting.
        left (Node): Left child node.
        right (Node): Right child node.
        value (int): Predicted class label for a leaf node.
    """
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """
        Checks if the node is a leaf node.

        Returns:
            bool: True if the node is a leaf node, False otherwise.
        """
        return self.value is not None

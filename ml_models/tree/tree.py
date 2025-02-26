import numpy as np
from queue import LifoQueue
from tree.impurity import IMPURITY_FNS

class Node:
    def __init__(self, impurity) -> None:
        self.impurity = impurity
        self.left_child = None
        self.right_child = None
        self.feature = None
        self.value = None

    def set_as_leaf(self, predicted_class):
        self.left_child = None
        self.right_child = None
        self.predicted_class = predicted_class

    def set_as_node(self, left_child, right_child, feature, value):
        self.left_child = left_child
        self.right_child = right_child
        self.feature = feature
        self.value = value


class Tree:
    def __init__(self, impurity="entropy") -> None:
        self.nodes = []
        self.impurity = impurity

    def fit(self, X, y):
        root_node = Node(impurity=self.compute_impurity(y))
        self.nodes.append(root_node)
        nodes_queue = LifoQueue()
        nodes_queue.put((X, y, root_node))

        while not nodes_queue.empty():
            X, y, node = nodes_queue.get()

            if node.impurity < 1e-9:  # if node is pure, then it's a leaf
                classes, counts = np.unique(y, return_counts=True)
                predicted_class = classes[np.argsort(counts)[-1]]
                self.set_as_leaf(node, predicted_class)
                continue

            # Split the node
            feature, value = self.find_splitting_criterion(node.impurity, X, y)
            _, left_child_impurity, right_child_impurity = self.compute_gain(
                node.impurity, X, y, feature, value, return_children_impurities=True
            )

            self.set_as_node(node, feature, value)

            # Create left and right child nodes
            left_child = Node(impurity=left_child_impurity)
            right_child = Node(impurity=right_child_impurity)
            self.nodes.extend([left_child, right_child])

            # Split the data and add to the queue
            mask = X[:, feature] <= value
            nodes_queue.put((X[mask], y[mask], left_child))
            nodes_queue.put((X[~mask], y[~mask], right_child))
        
        return self

    def compute_impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return IMPURITY_FNS[self.impurity](p)

    def compute_gain(self, node_impurity, X, y, feature, value, return_children_impurities=False):
        mask = X[:, feature] <= value
        y_left, y_right = y[mask], y[~mask]
        left_child_impurity = self.compute_impurity(y_left)
        right_child_impurity = self.compute_impurity(y_right)

        gain = node_impurity - (y_left.shape[0] * left_child_impurity + y_right.shape[0] * right_child_impurity) / y.shape[0]

        if return_children_impurities:
            return gain, left_child_impurity, right_child_impurity

        return gain

    def find_splitting_criterion(self, node_impurity, X, y):
        max_info_gain = -float("inf")
        splitting_feature = 0
        splitting_value = X[0, 0]

        for feature in range(X.shape[1]): # this can be improved to avoid redudant computations i f working with categorical data
            for sample in range(X.shape[0]):
                value = X[sample, feature]
                info_gain = self.compute_gain(node_impurity, X, y, feature, value)

                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    splitting_feature = feature
                    splitting_value = value

        return splitting_feature, splitting_value

    def set_as_leaf(self, node, predicted_class):
        node.set_as_leaf(predicted_class)

    def set_as_node(self, node, feature, value):
        left_child = len(self.nodes)
        right_child = len(self.nodes) + 1
        node.set_as_node(left_child=left_child, right_child=right_child, feature=feature, value=value)

### I didn't code this

    def predict(self, X):
        predictions = []
        for sample in X:
            predictions.append(self._predict_sample(sample))
        
        return np.array(predictions)
    
    def _predict_sample(self, sample):
        node = self.nodes[0]
        
        while node.left_child is not None and node.right_child is not None:            
            feature_value = sample[node.feature]
            if feature_value <= node.value:
                node = self.nodes[node.left_child]
            else:
                node = self.nodes[node.right_child]

        return node.predicted_class  

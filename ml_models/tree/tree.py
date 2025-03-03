import numpy as np
from queue import LifoQueue
from tree.impurity import IMPURITY_FNS
from typing import Tuple


class Node:
    def __init__(self, impurity: float) -> None:
        self.impurity: float = impurity
        self.left_child: int | None = None
        self.right_child: int | None = None
        self.feature: int | None = None
        self.value: float | None = None
        self.predicted_class: int | None = None

    def set_as_leaf(self, predicted_class: int) -> None:
        self.left_child = None
        self.right_child = None
        self.predicted_class = predicted_class

    def set_as_node(
        self,
        left_child: int,
        right_child: int,
        feature: int,
        value: float,
    ) -> None:
        self.left_child = left_child
        self.right_child = right_child
        self.feature = feature
        self.value = value


class Tree:
    def __init__(self, impurity: str = "entropy") -> None:
        self.nodes: list[Node] = []
        self.impurity: str = impurity

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Tree":
        root_node = Node(impurity=self.compute_impurity(y))
        self.nodes.append(root_node)
        nodes_queue: LifoQueue[Tuple[np.ndarray, np.ndarray, Node]] = (
            LifoQueue()
        )
        nodes_queue.put((X, y, root_node))

        while not nodes_queue.empty():
            X, y, node = nodes_queue.get()

            if node.impurity < 1e-9:  # if node is pure, then it's a leaf
                classes, counts = np.unique(y, return_counts=True)
                predicted_class = classes[np.argmax(counts)]
                self.set_as_leaf(node, int(predicted_class))
                continue

            feature, value = self.find_splitting_criterion(node.impurity, X, y)
            _, left_child_impurity, right_child_impurity = self.compute_gain(
                node.impurity,
                X,
                y,
                feature,
                value,
                return_children_impurities=True,
            )

            self.set_as_node(node, feature, value)

            left_child = Node(impurity=left_child_impurity)
            right_child = Node(impurity=right_child_impurity)
            self.nodes.extend([left_child, right_child])

            mask = X[:, feature] <= value
            nodes_queue.put((X[mask], y[mask], left_child))
            nodes_queue.put((X[~mask], y[~mask], right_child))

        return self

    def compute_impurity(self, y: np.ndarray) -> float:
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return IMPURITY_FNS[self.impurity](p)

    def compute_gain(
        self,
        node_impurity: float,
        X: np.ndarray,
        y: np.ndarray,
        feature: int,
        value: float,
        return_children_impurities: bool = False,
    ) -> float | Tuple[float, float, float]:
        mask = X[:, feature] <= value
        y_left, y_right = y[mask], y[~mask]
        left_child_impurity = self.compute_impurity(y_left)
        right_child_impurity = self.compute_impurity(y_right)

        gain = (
            node_impurity
            - (
                y_left.shape[0] * left_child_impurity
                + y_right.shape[0] * right_child_impurity
            )
            / y.shape[0]
        )

        if return_children_impurities:
            return gain, left_child_impurity, right_child_impurity

        return gain

    def find_splitting_criterion(
        self, node_impurity: float, X: np.ndarray, y: np.ndarray
    ) -> Tuple[int, float]:
        max_info_gain = -float("inf")
        splitting_feature = 0
        splitting_value = X[0, 0]

        for feature in range(X.shape[1]):
            for sample in range(X.shape[0]):
                value = X[sample, feature]
                info_gain = self.compute_gain(
                    node_impurity, X, y, feature, value
                )

                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    splitting_feature = feature
                    splitting_value = value

        return splitting_feature, splitting_value

    def set_as_leaf(self, node: Node, predicted_class: int) -> None:
        node.set_as_leaf(predicted_class)

    def set_as_node(self, node: Node, feature: int, value: float) -> None:
        left_child = len(self.nodes)
        right_child = len(self.nodes) + 1
        node.set_as_node(
            left_child=left_child,
            right_child=right_child,
            feature=feature,
            value=value,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = [self._predict_sample(sample) for sample in X]
        return np.array(predictions)

    def _predict_sample(self, sample: np.ndarray) -> int:
        node = self.nodes[0]

        while node.left_child is not None and node.right_child is not None:
            feature_value = sample[node.feature]
            if feature_value <= node.value:
                node = self.nodes[node.left_child]
            else:
                node = self.nodes[node.right_child]

        return node.predicted_class

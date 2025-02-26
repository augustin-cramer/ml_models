import numpy as np

def entropy(p):
    return -np.sum(p * np.log(p))


IMPURITY_FNS = {
    "entropy": entropy
}
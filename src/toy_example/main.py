#!/usr/bin/env python3
import torch
import torch.nn as nn


def feature_cost_mapping(w):
    """
    Docstring for feature_cost_mapping

    :param w: Description
    :return: Description
    """
    # TODO: implement mapping for data batches, i.e., when w \in R^{1xm}
    if w >= 0:
        c = [w, w, 2*w, 2*w]
    else:
        c = [-2*w, -2*w, -w, -w]
    return c

def optimizer(c):
    """
    Docstring for optimizer
    
    :param c: Description
    :return: Description
    """
    if c[0] + c[1] <= c[2] + c[3]:
        return [1, 1, 0, 0]
    else:
        return [0, 0, 1, 1]
    
def interdictor(c):
    """
    Docstring for interdictor
    
    :param c: Description
    :return: Description
    """
    if c[0] + c[1] <= c[2] + c[3]:
        return [0, 3, 0, 0]
    else:
        return [0, 0, 0, 3]
    
def test_intd_pipeline_volatile():
    w = -1
    print(f"Input                w:  {w}")
    c = feature_cost_mapping(w)
    print(f"Cost                 c: {c}")
    y = optimizer(c)
    print(f"Opt. sol.            y: {y}")
    g_c = interdictor(c)
    print(f"Intd.              g_c: {g_c}")
    c_intd = [c[i] + g_c[i] for i in range(len(c))]
    print(f"Intd. cost     c + g_c: {c_intd}")
    y_intd = optimizer(c_intd)
    print(f"Intd. opt. sol. y_intd: {y_intd}")
    pass

def test_toy_example():
    predictor = nn.Sequential(
        nn.Linear(4, 1),
        nn.ReLU(),
        nn.Linear(4, 4)
    )
    # TODO: implement training and testing loops
    pass

def train_dfl_predictor():
    pass

def train_adfl_predictor():
    pass

def main():
    test_toy_example()

if __name__ == "__main__":
    main()

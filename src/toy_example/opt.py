#!/usr/bin/env python3

from toy_example.main_funcs import optimizer, sol_value
from pyepo.model.opt import optModel

class ToyOptModel(optModel):
    def __init__(self, cost):
        self.cost = cost
        super().__init__()

    def _getModel(self):
        model = self.cost
        x = None
        return model, x

    def setObj(self, c):
        self.cost = c

    def solve(self):
        y = optimizer(self.cost)
        obj_val = sol_value(self.cost, y)
        return y, obj_val
        



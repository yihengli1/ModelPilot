import numpy as np

"""
Manual classes from school assignment lol
"""


class LearningRateGetter:
    def __init__(self, multiplier):
        self.multiplier = multiplier
        self.num_evals = 0

    def reset(self):
        self.num_evals = 0

    def get_learning_rate(self):
        raise NotImplementedError


class ConstantLR(LearningRateGetter):
    def get_learning_rate(self):
        self.num_evals += 1
        return self.multiplier


class InverseLR(LearningRateGetter):
    def get_learning_rate(self):
        self.num_evals += 1
        return self.multiplier / self.num_evals


class InverseSquaredLR(LearningRateGetter):
    def get_learning_rate(self):
        self.num_evals += 1
        return self.multiplier / (self.num_evals ** 2)


class InverseSqrtLR(LearningRateGetter):
    def get_learning_rate(self):
        self.num_evals += 1
        return self.multiplier / np.sqrt(self.num_evals)

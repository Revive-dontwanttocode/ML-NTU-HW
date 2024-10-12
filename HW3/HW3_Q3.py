"""
A calculator of question3 in homework3
"""

import math
import numpy as np

w1 = [-1, 0, 0, 0.5, 0, -0.5]
w2 = [-1, 0, 0, -0.5, 0, 0.5]
w3 = [-2, 0, 0, 1, 0, 1]
w4 = [-1, 0, 0, 0.2, 0, 0.1]
x1 = [0, 0]
y1 = -1

x2 = [4, 0]
y2 = 1

x3 = [-4, 0]
y3 = 1

x4 = [0, 2]
y4 = -1

x5 = [0, -2]
y5 = -1


def z_space_transform(vector):
    new_coordinate = [1, vector[0], vector[1], vector[0] ** 2, vector[0] * vector[1], vector[1] ** 2]
    return new_coordinate

def training_func(input_vector, weight):
    return 1 if (sum(x * y for x, y in zip(z_space_transform(input_vector), weight)) >= 0) else -1


print(training_func(x1, w1))
print(training_func(x2, w1))
print(training_func(x3, w1))
print(training_func(x4, w1))
print(training_func(x5, w1))

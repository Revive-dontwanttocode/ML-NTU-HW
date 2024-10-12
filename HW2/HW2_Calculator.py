"""
Calculator for homework 02
"""

import math


def calculate_bad_probability(N):
    return 16 * (N ** 2) * (math.e ** ((-1 / 8) * math.pow(1 / 20, 2) * N))


for i in [100, 1000, 10000, 100000, 1000000]:
    print(f"i = {i}, outcome = {calculate_bad_probability(i)}")
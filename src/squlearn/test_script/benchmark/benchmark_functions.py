from scipy.optimize import rosen
import numpy as np

"""
- Dimension: n
- Minimum: f(0,0,...,0)=0
"""


def sphere_function(x):
    return sum(xi ** 2 for xi in x)


"""
- Dimension: n
- Minimum: f(1,1,...,1)=0
"""


def rosenbrock_function(x):
    return rosen(x)


"""
- Dimension: n
- Minimum: f(0,0,...,0)=0
"""


def ackley_function(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / len(x))) - np.exp(
        np.sum(np.cos(2 * np.pi * x)) / len(x)) + 20 + np.exp(1)


"""
- Dimension: n
- Minimum: f(0,0,...,0)=0
"""


def griewank_function(x):
    return 1 / 4000 * sum(xi ** 2 for xi in x) - np.prod(np.cos(xi / np.sqrt(i + 1)) for i, xi in enumerate(x)) + 1


"""
- Dimension: n
- Minimum: f(0,0,...,0)=0
"""


def rastrigin_function(x):
    return 10 * len(x) + sum(xi ** 2 - 10 * np.cos(2 * np.pi * xi) for xi in x)


"""
- Dimension: 2
- Minimum: Vier identische globale Minima bei (3,2),(−2.805118,3.131312),(−3.779310,−3.283186),(3.584428,−1.848126)
"""


def himmelblau_function(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


"""
- Dimension: 2
- Minimum: f(0,0)=0
"""


def schaffer_function(x, y):
    return 0.5 + ((np.sin(x ** 2 - y ** 2) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + y ** 2) ** 2))


"""
- Dimension: 2
- Minimum: f(π,π)=−1
"""


def easom_function(x, y):
    return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))


'''
- Dimension: 
- Minimum: Anzahl der globale Minima variiert mit m und n, typischerweise viele lokale Minima
'''


def michalewicz_function(x, m=10):
    return -sum(np.sin(xi) * np.sin((i + 1) * xi ** 2 / np.pi) ** (2 * m) for i, xi in enumerate(x))


'''
- Dimension: 2
- Minimum: f(1,3)=0
'''


def booth_function(x, y):
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

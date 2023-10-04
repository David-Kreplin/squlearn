"""A library for quantum machine learning following the scikit-learnstandard."""
from .util import Executor
from . import observable, feature_map, kernel, optimizers, qnn, util

__version__ = "0.3.0"

__all__ = [
    "Executor",
    "observable",
    "feature_map",
    "kernel",
    "optimizers",
    "qnn",
    "util",
]

from collections import namedtuple

from numba import vectorize, float32, float64
import numpy as np

Interval = namedtuple("Interval", ["lo", "hi"])

def second_moment(X):
    n = X.shape[0]
    return np.transpose(X).dot(X) / n

def cov(X):
    mean = np.mean(X, axis = 0)
    return second_moment(X) - np.outer(mean, mean)

@vectorize([float32(float32), float64(float64)])
def threshold(x):
    return float(x > 0.0)


def clip(interval):
    return intersect([interval, Interval(0.0, 1.0)])


def intersect(intervals):
    return Interval(lo=np.max([_.lo for _ in intervals]),
                    hi=np.min([_.hi for _ in intervals]))


def get_grid(interval, num_points=100):
    return np.linspace(interval.lo, interval.hi, num=num_points)
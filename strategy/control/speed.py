from itertools import islice
from operator import mul
from functools import reduce
from strategy.common import Point


def get_speed(position: Point, direction: Point, path):
    if len(path) < 1:
        return direction * 100
    path = [position] + path

    def generate_cos():
        for i, current in islice(enumerate(path), 1, min(3, len(path) - 1)):
            yield (current - path[i - 1]).cos(path[i + 1] - current)

    return (path[1] - path[0]) * speed_gain(reduce(mul, generate_cos(), 1))


def speed_gain(x):
    return 1 - 3 / (x - 1)

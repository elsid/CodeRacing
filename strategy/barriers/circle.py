from numpy import sign
from scipy.optimize import bisect
from strategy.common import Line


class Circle:
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius

    def __repr__(self):
        return 'Circle(position={p}, radius={r})'.format(
            p=repr(self.position), r=repr(self.radius))

    def __eq__(self, other):
        return (self.position == other.position and
                self.radius == other.radius)

    def passability(self, position, radius, _=None):
        distance = (self.position - position).norm()
        return float(distance > self.radius + radius)

    def intersection_with_line(self, line: Line):
        nearest = line.nearest(self.position)
        distance = self.position.distance(nearest)
        if (distance > self.radius or
                (line.begin - nearest).dot(line.end - nearest) > 0):
            return []
        if self.radius == distance:
            return [nearest]

        def generate():
            to_begin = Line(nearest, line.begin)
            if to_begin.length() > 0:
                def func(parameter):
                    return (self.position.distance(to_begin(parameter)) -
                            self.radius)
                if sign(func(0)) != sign(func(1)):
                    yield to_begin(bisect(func, 0, 1))
            to_end = Line(nearest, line.end)
            if to_end.length() > 0:
                def func(parameter):
                    return (self.position.distance(to_end(parameter)) -
                            self.radius)
                if sign(func(0)) != sign(func(1)):
                    yield to_end(bisect(func, 0, 1))

        return list(generate())

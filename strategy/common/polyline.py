from itertools import islice
from strategy.common import Line


class Polyline:
    def __init__(self, points):
        self.points = points

    def distance(self, point):
        points = islice(enumerate(self.points), len(self.points) - 1)
        return min(Line(p, self.points[i - 1]).nearest(point).distance(point)
                   for i, p in points)

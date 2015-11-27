from strategy.common import Point


class Line:
    def __init__(self, begin: Point, end: Point):
        self.begin = begin
        self.end = end

    def __call__(self, parameter):
        return self.begin + (self.end - self.begin) * parameter

    def distance(self, point):
        to_end = self.end - self.begin
        to_point = point - self.begin
        norm = to_point.dot(to_end) / to_end.norm()
        return sqrt(to_point.norm() ** 2 - norm ** 2)

    def nearest(self, point):
        to_end = self.end - self.begin
        to_point = point - self.begin
        norm = to_point.dot(to_end) / to_end.norm()
        return self.begin + to_end / to_end.norm() * norm

    def length(self):
        return (self.end - self.begin).norm()

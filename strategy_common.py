from collections import deque
from itertools import islice
from math import cos, sin, sqrt, atan2, pi
from numpy import arctan2
from json import dumps
from scipy.interpolate import InterpolatedUnivariateSpline


def get_current_tile(point, tile_size):
    return Point(tile_coord(point.x, tile_size), tile_coord(point.y, tile_size))


def tile_coord(value, tile_size):
    return int(value / tile_size)


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return 'Point(x={x}, y={y})'.format(x=self.x, y=self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        else:
            return Point(self.x + other, self.y + other)

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        else:
            return Point(self.x - other, self.y - other)

    def __mul__(self, other):
        if isinstance(other, Point):
            return Point(self.x * other.x, self.y * other.y)
        else:
            return Point(self.x * other, self.y * other)

    def __truediv__(self, other):
        if isinstance(other, Point):
            return Point(self.x / other.x, self.y / other.x)
        else:
            return Point(self.x / other, self.y / other)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __lt__(self, other):
        if self.x != other.x:
            return self.x < other.x
        return self.y < other.y

    @property
    def radius(self):
        return self.x

    @radius.setter
    def radius(self, value):
        self.x = value

    @property
    def angle(self):
        return self.y

    @angle.setter
    def angle(self, value):
        self.y = value

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def norm(self):
        return sqrt(self.dot(self))

    def cos(self, other):
        return self.dot(other) / (self.norm() * other.norm())

    def distance(self, other):
        return (other - self).norm()

    def map(self, function):
        return Point(function(self.x), function(self.y))

    def polar(self, cartesian_origin=None):
        if cartesian_origin:
            return (self - cartesian_origin).polar()
        else:
            radius = self.norm()
            angle = arctan2(self.y, self.x)
            return Point(radius, angle)

    def cartesian(self, cartesian_origin=None):
        if cartesian_origin:
            return self.cartesian() + cartesian_origin
        else:
            return Point(x=self.radius * cos(self.angle),
                         y=self.radius * sin(self.angle))

    def left_orthogonal(self):
        return Point(-self.y, self.x)

    def absolute_rotation(self):
        return atan2(self.y, self.x)

    def rotation(self, other):
        return other.absolute_rotation() - self.absolute_rotation()

    def rotate(self, angle):
        return Point(self.x * cos(angle) - self.y * sin(angle),
                     self.y * cos(angle) + self.x * sin(angle))

    def normalized(self):
        return self / self.norm()

    def projection(self, other):
        return other * self.dot(other) / other.norm()

    def manhattan(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)


class Line:
    def __init__(self, begin: Point, end: Point):
        self.begin = begin
        self.end = end

    def __call__(self, parameter):
        return self.begin + (self.end - self.begin) * parameter

    def __repr__(self):
        return 'Line(begin={b}, end={e})'.format(
            b=repr(self.begin), e=repr(self.end))

    def distance(self, point):
        to_end = self.end - self.begin
        to_point = point - self.begin
        norm = to_point.dot(to_end) / to_end.norm()
        return sqrt(to_point.norm() ** 2 - norm ** 2)

    def nearest(self, point):
        to_end = self.end - self.begin
        to_end_norm = to_end.norm()
        if to_end_norm == 0:
            return Point(self.begin.x, self.begin.y)
        to_point = point - self.begin
        return self.begin + to_end * to_point.dot(to_end) / (to_end_norm ** 2)

    def length(self):
        return (self.end - self.begin).norm()

    def has_point(self, point: Point, max_error=1e-8):
        to_end = self.end - point
        if to_end.norm() == 0:
            return True
        to_begin = self.begin - point
        if to_begin.norm() == 0:
            return True
        return abs(1 + to_begin.cos(to_end)) <= max_error

    def __eq__(self, other):
        return self.begin == other.begin and self.end == other.end

    def intersection(self, other):
        x_diff = Point(self.begin.x - self.end.x, other.begin.x - other.end.x)
        y_diff = Point(self.begin.y - self.end.y, other.begin.y - other.end.y)

        def det(a, b):
            return a.x * b.y - a.y * b.x

        div = det(x_diff, y_diff)
        if div == 0:
            return None
        d = Point(det(self.begin, self.end), det(other.begin, other.end))
        x = det(d, x_diff) / div
        y = det(d, y_diff) / div
        return Point(x, y)


class Polyline:
    def __init__(self, points):
        assert points
        self.points = points

    def nearest_point(self, point):
        if len(self.points) < 2:
            return self.points[0]
        points = islice(enumerate(self.points), len(self.points) - 1)
        nearest = (Line(p, self.points[i - 1]).nearest(point)
                   for i, p in points)
        return min(nearest, key=lambda p: p.distance(point))

    def distance(self, point):
        return self.nearest_point(point).distance(point)

    def at(self, distance):
        for i, p in islice(enumerate(self.points), len(self.points) - 1):
            to_next = self.points[i + 1] - p
            to_next_distance = to_next.norm()
            if to_next_distance < distance:
                distance -= to_next_distance
            else:
                return p + to_next.normalized() * distance
        return self.points[-1]

    def length(self):
        if not self.points:
            return 0
        result = 0
        for i, p in islice(enumerate(self.points), len(self.points) - 1):
            result += p.distance(self.points[i + 1])
        return result


def get_tile_center(point: Point, size):
    return point.map(lambda x: tile_center_coord(x, size))


def tile_center_coord(value, size):
    return (value + 0.5) * size


def normalize_angle(value):
    if value > pi:
        return value - round(value / (2.0 * pi)) * 2.0 * pi
    if value < -pi:
        return value + round(abs(value) / (2.0 * pi)) * 2.0 * pi
    return value


class LimitedSum:
    def __init__(self, history_size):
        self.__values = deque(maxlen=history_size)
        self.__amount = 0

    def update(self, value):
        first = (self.__values[0] if len(self.__values) == self.__values.maxlen
                 else None)
        self.__values.append(value)
        self.__amount += value - (0 if first is None else first)

    def reset(self):
        self.__values.clear()
        self.__amount = 0

    def get(self):
        return self.__amount

    @property
    def count(self):
        return len(self.__values)

    @property
    def max_count(self):
        return self.__values.maxlen


class Curve:
    def __init__(self, points):
        begin = points[0]
        axis = points[-1] - begin
        angle = axis.absolute_rotation()
        relative = ((p - begin).rotate(-angle) for p in points)
        polar = [p.polar() for p in relative]
        x = [p.radius for p in polar]
        y = [p.angle for p in polar]
        self.__spline = InterpolatedUnivariateSpline(x, y, k=min(5, len(x) - 1))
        self.__angle = angle
        self.__begin = begin

    def at(self, distance):
        angle = self.__spline(distance)
        point = Point(distance, angle)
        return self.__begin + point.cartesian().rotate(self.__angle)

from itertools import islice
from math import acos, cos, sin, sqrt, atan2, pi
from numpy import arctan2, sign


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

    def __iadd__(self, other):
        if isinstance(other, Point):
            self.x += other.x
            self.y += other.y
        else:
            self.x += other
            self.y += other
        return self

    def __isub__(self, other):
        if isinstance(other, Point):
            self.x -= other.x
            self.y -= other.y
        else:
            self.x -= other
            self.y -= other
        return self

    def __imul__(self, other):
        if isinstance(other, Point):
            self.x *= other.x
            self.y *= other.y
        else:
            self.x *= other
            self.y *= other
        return self

    def __itruediv__(self, other):
        if isinstance(other, Point):
            self.x /= other.x
            self.y /= other.y
        else:
            self.x /= other
            self.y /= other
        return self

    def __neg__(self):
        return Point(-self.x, -self.y)

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


class Polyline:
    def __init__(self, points):
        self.points = points

    def distance(self, point):
        points = islice(enumerate(self.points), len(self.points) - 1)
        return min(Line(p, self.points[i - 1]).nearest(point).distance(point)
                   for i, p in points)


def get_tile_center(point: Point, size):
    return point.map(lambda x: tile_center_coord(x, size))


def tile_center_coord(value, size):
    return (value + 0.5) * size

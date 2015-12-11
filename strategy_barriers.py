from model.CircularUnit import CircularUnit
from model.RectangularUnit import RectangularUnit
from model.TileType import TileType
from numpy import sign
from scipy.optimize import bisect
from strategy_common import Point, Line
from strategy_path import get_point_index, get_current_tile


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

    def has_intersection_with_line(self, line: Line):
        nearest = line.nearest(self.position)
        if nearest.distance(self.position) > self.radius:
            return False
        return line.has_point(nearest)

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


def make_passability_function(barriers, radius, speed, tiles, tile_size):
    def impl(x, y):
        tile = get_current_tile(Point(x, y), tile_size)
        if tile not in tiles:
            return 0.0
        return min((b.passability(Point(x, y), radius, speed)
                    for b in barriers), default=1.0)
    return impl


class Rectangle:
    INSIDE = 0
    LEFT = 1
    RIGHT = 2
    TOP = 4
    BOTTOM = 8

    def __init__(self, left_top, right_bottom):
        self.left_top = left_top
        self.right_bottom = right_bottom

    def __repr__(self):
        return 'Rectangle(left_top={lt}, right_bottom={rb})'.format(
            lt=repr(self.left_top), rb=repr(self.right_bottom))

    def __eq__(self, other):
        return (self.left_top == other.left_top and
                self.right_bottom == other.right_bottom)

    def passability(self, position, radius, _=None):
        position_code = self.point_code(position)
        if position_code == Rectangle.INSIDE:
            return 0.0
        width = self.right_bottom.x - self.left_top.x
        height = self.right_bottom.y - self.left_top.y
        center = self.left_top + Point(width / 2, height / 2)
        direction = center - position
        border = position + direction / direction.norm() * radius
        border_code = self.point_code(border)
        return float(position_code & border_code)

    def point_code(self, point):
        result = Rectangle.INSIDE
        if point.x < self.left_top.x:
            result |= Rectangle.LEFT
        elif point.x > self.right_bottom.x:
            result |= Rectangle.RIGHT
        if point.y < self.left_top.y:
            result |= Rectangle.TOP
        elif point.y > self.right_bottom.y:
            result |= Rectangle.BOTTOM
        return result

    def left(self):
        return Line(begin=self.left_top,
                    end=self.left_top + Point(0, self.height()))

    def right(self):
        return Line(begin=self.right_bottom,
                    end=self.right_bottom - Point(0, self.height()))

    def top(self):
        return Line(begin=self.left_top + Point(self.width(), 0),
                    end=self.left_top)

    def bottom(self):
        return Line(begin=self.right_bottom - Point(self.width(), 0),
                    end=self.right_bottom)

    def width(self):
        return self.right_bottom.x - self.left_top.x

    def height(self):
        return self.right_bottom.y - self.left_top.y

    def clip_line(self, line: Line):
        k1 = self.point_code(line.begin)
        k2 = self.point_code(line.end)
        x1 = line.begin.x
        y1 = line.begin.y
        x2 = line.end.x
        y2 = line.end.y
        left = self.left_top.x
        top = self.left_top.y
        right = self.right_bottom.x
        bottom = self.right_bottom.y
        accept = False
        while True:
            if (k1 | k2) == 0:
                accept = True
                break
            if (k1 & k2) != 0:
                break
            opt = k1 or k2
            if opt & Rectangle.TOP:
                x = x1 + (x2 - x1) * (bottom - y1) / (y2 - y1)
                y = bottom
            elif opt & Rectangle.BOTTOM:
                x = x1 + (x2 - x1) * (top - y1) / (y2 - y1)
                y = top
            if opt & Rectangle.RIGHT:
                y = y1 + (y2 - y1) * (right - x1) / (x2 - x1)
                x = right
            elif opt & Rectangle.LEFT:
                y = y1 + (y2 - y1) * (left - x1) / (x2 - x1)
                x = left
            if opt == k1:
                x1, y1 = x, y
                k1 = self.point_code(Point(x1, y1))
            elif opt == k2:
                x2, y2 = x, y
                k2 = self.point_code(Point(x2, y2))
        if accept:
            return Line(Point(x1, y1), Point(x2, y2))
        else:
            return line

    def has_intersection_with_line(self, line: Line):
        return line != self.clip_line(line)


def make_tiles_barriers(tiles, margin, size):
    row_size = len(tiles[0])

    def generate():
        for x, column in enumerate(tiles):
            for y, tile in enumerate(column):
                position = Point(x, y)
                yield get_point_index(position, row_size), make_tile_barriers(
                    tile_type=tile,
                    position=position,
                    margin=margin,
                    size=size,
                )

    return dict(generate())


def make_tile_barriers(tile_type: TileType, position: Point, margin, size):
    absolute_position = position * size

    def point(x, y):
        return absolute_position + Point(x, y)

    left = Rectangle(left_top=point(0, 0), right_bottom=point(margin, size))
    right = Rectangle(left_top=point(size - margin, 0),
                      right_bottom=point(size, size))
    top = Rectangle(left_top=point(0, 0), right_bottom=point(size, margin))
    bottom = Rectangle(left_top=point(0, size - margin),
                       right_bottom=point(size, size))
    left_top = Circle(point(0, 0), margin)
    left_bottom = Circle(point(0, size), margin)
    right_top = Circle(point(size, 0), margin)
    right_bottom = Circle(point(size, size), margin)
    if tile_type == TileType.VERTICAL:
        return [left, right]
    elif tile_type == TileType.HORIZONTAL:
        return [top, bottom]
    elif tile_type == TileType.LEFT_TOP_CORNER:
        return [left, top, right_bottom]
    elif tile_type == TileType.RIGHT_TOP_CORNER:
        return [right, top, left_bottom]
    elif tile_type == TileType.LEFT_BOTTOM_CORNER:
        return [left, bottom, right_top]
    elif tile_type == TileType.RIGHT_BOTTOM_CORNER:
        return [right, bottom, left_top]
    elif tile_type == TileType.LEFT_HEADED_T:
        return [left_top, left_bottom, right]
    elif tile_type == TileType.RIGHT_HEADED_T:
        return [right_top, right_bottom, left]
    elif tile_type == TileType.TOP_HEADED_T:
        return [left_top, right_top, bottom]
    elif tile_type == TileType.BOTTOM_HEADED_T:
        return [left_bottom, right_bottom, top]
    elif tile_type == TileType.CROSSROADS:
        return [left_top, left_bottom, right_top, right_bottom]
    else:
        return [Rectangle(left_top=point(0, 0), right_bottom=point(size, size))]


class Unit:
    def __init__(self, barrier, speed):
        self.barrier = barrier
        self.speed = speed

    def has_intersection_with_line(self, line: Line):
        return self.barrier.has_intersection_with_line(line)

    def __repr__(self):
        return 'Unit(barrier={b}, speed={s})'.format(
            b=repr(self.barrier), s=repr(self.speed))

    def __eq__(self, other):
        return (self.barrier == other.__barrier and
                self.speed == other.__speed)


def make_units_barriers(units):
    return [x for x in (make_unit_barrier(x) for x in units) if x]


def make_unit_barrier(unit):
    if isinstance(unit, RectangularUnit):
        return Unit(
            barrier=Circle(position=Point(unit.x, unit.y),
                           radius=min(unit.width, unit.height) / 2),
            speed=Point(unit.speed_x, unit.speed_y),
        )
    elif isinstance(unit, CircularUnit):
        return Unit(
            barrier=Circle(position=Point(unit.x, unit.y), radius=unit.radius),
            speed=Point(unit.speed_x, unit.speed_y),
        )
    else:
        return None



def make_has_intersection_with_line(position, course, barriers):
    def impl(angle):
        end = position + course.rotate(angle)
        line = Line(position, end)
        return next((True for x in barriers
                     if x.has_intersection_with_line(line)), False)
    return impl


def make_has_intersection_with_lane(position, course, barriers, width):
    orthogonal = course.left_orthogonal().normalized() * width / 2
    left = position + orthogonal
    right = position - orthogonal
    left_has_intersection = make_has_intersection_with_line(
        position=left,
        course=course,
        barriers=barriers,
    )
    middle_has_intersection = make_has_intersection_with_line(
        position=position,
        course=course,
        barriers=barriers,
    )
    right_has_intersection = make_has_intersection_with_line(
        position=right,
        course=course,
        barriers=barriers,
    )

    def impl(angle):
        return (left_has_intersection(angle) or
                middle_has_intersection(angle) or
                right_has_intersection(angle))

    return impl

from collections import namedtuple
from itertools import islice
from enum import Enum
from strategy.common import Point


def adjust_path(path, tile_size):
    if len(path) < 2:
        return (x for x in path)

    def generate():
        typed_path = list(make_typed_path())
        for i, p in islice(enumerate(typed_path), 0, len(typed_path) - 1):
            yield adjust_path_point(p, typed_path[i + 1], tile_size)
        yield path[-1]

    def make_typed_path():
        yield TypedPoint(path[0],
                         PointType(SideType.UNKNOWN,
                                   output_type(path[0], path[1])))
        for i, p in islice(enumerate(path), 1, len(path) - 1):
            yield TypedPoint(p, point_type(path[i - 1], p, path[i + 1]))
        yield TypedPoint(path[-1],
                         PointType(input_type(path[-2], path[-1]),
                                   SideType.UNKNOWN))

    return generate()


TypedPoint = namedtuple('TypedPoint', ('position', 'type'))


def adjust_path_point(current: TypedPoint, following: TypedPoint,
                      tile_size):
    if current.type == following.type:
        return current.position
    shift = Point(0, 0)
    if current.type in {PointType.LEFT_TOP, PointType.TOP_LEFT}:
        shift = Point(- tile_size / 4, - tile_size / 4)
        if current.type.input == following.type.output:
            shift /= 2
    elif current.type in {PointType.LEFT_BOTTOM, PointType.BOTTOM_LEFT}:
        shift = Point(- tile_size / 4, + tile_size / 4)
        if current.type.input == following.type.output:
            shift /= 2
    elif current.type in {PointType.RIGHT_TOP, PointType.TOP_RIGHT}:
        shift = Point(+ tile_size / 4, - tile_size / 4)
        if current.type.input == following.type.output:
            shift /= 2
    elif current.type in {PointType.RIGHT_BOTTOM, PointType.BOTTOM_RIGHT}:
        shift = Point(+ tile_size / 4, + tile_size / 4)
        if current.type.input == following.type.output:
            shift /= 2
    elif current.type in {PointType.LEFT_RIGHT, PointType.RIGHT_LEFT}:
        if following.type.output == SideType.TOP:
            shift = Point(0, + tile_size / 4)
        else:
            shift = Point(0, - tile_size / 4)
    elif current.type in {PointType.TOP_BOTTOM, PointType.BOTTOM_TOP}:
        if following.type.output == SideType.LEFT:
            shift = Point(+ tile_size / 4, 0)
        else:
            shift = Point(- tile_size / 4, 0)
    return current.position + shift


def point_type(previous, current, following):
    return PointType(input_type(previous, current),
                     output_type(current, following))


def input_type(previous, current):
    if previous.y == current.y:
        if previous.x < current.x:
            return SideType.LEFT
        else:
            return SideType.RIGHT
    if previous.x == current.x:
        if previous.y < current.y:
            return SideType.TOP
        else:
            return SideType.BOTTOM
    return SideType.UNKNOWN


def output_type(current, following):
    if current.y == following.y:
        if current.x < following.x:
            return SideType.RIGHT
        else:
            return SideType.LEFT
    if current.x == following.x:
        if current.y < following.y:
            return SideType.BOTTOM
        else:
            return SideType.TOP
    return SideType.UNKNOWN


class SideType(Enum):
    UNKNOWN = 0
    LEFT = 1
    RIGHT = 2
    TOP = 3
    BOTTOM = 4


PointTypeImpl = namedtuple('PointType', ('input', 'output'))


class PointType(PointTypeImpl):
    LEFT_RIGHT = PointTypeImpl(SideType.LEFT, SideType.RIGHT)
    LEFT_TOP = PointTypeImpl(SideType.LEFT, SideType.TOP)
    LEFT_BOTTOM = PointTypeImpl(SideType.LEFT, SideType.BOTTOM)
    RIGHT_LEFT = PointTypeImpl(SideType.RIGHT, SideType.LEFT)
    RIGHT_TOP = PointTypeImpl(SideType.RIGHT, SideType.TOP)
    RIGHT_BOTTOM = PointTypeImpl(SideType.RIGHT, SideType.BOTTOM)
    TOP_LEFT = PointTypeImpl(SideType.TOP, SideType.LEFT)
    TOP_RIGHT = PointTypeImpl(SideType.TOP, SideType.RIGHT)
    TOP_BOTTOM = PointTypeImpl(SideType.TOP, SideType.BOTTOM)
    BOTTOM_LEFT = PointTypeImpl(SideType.BOTTOM, SideType.LEFT)
    BOTTOM_RIGHT = PointTypeImpl(SideType.BOTTOM, SideType.RIGHT)
    BOTTOM_TOP = PointTypeImpl(SideType.BOTTOM, SideType.TOP)

from model.TileType import TileType
from strategy.common import Point
from strategy.barriers.rectangle import Rectangle
from strategy.barriers.circle import Circle


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
        return []

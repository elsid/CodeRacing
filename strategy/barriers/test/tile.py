from unittest import TestCase
from hamcrest import assert_that, equal_to
from model.TileType import TileType
from strategy.common import Point
from strategy.barriers import make_tile_barriers
from strategy.barriers.circle import Circle
from strategy.barriers.rectangle import Rectangle


class MakeTileBarriersTest(TestCase):
    def test_for_empty_returns_empty_list(self):
        result = make_tile_barriers(tile_type=TileType.EMPTY,
                                    position=Point(10, 10), margin=1, size=3)
        assert_that(result, equal_to([]))

    def test_for_vertical_returns_two_rectangles(self):
        result = make_tile_barriers(tile_type=TileType.VERTICAL,
                                    position=Point(10, 20), margin=1, size=3)
        assert_that(result, equal_to([
            Rectangle(left_top=Point(30, 60), right_bottom=Point(31, 63)),
            Rectangle(left_top=Point(32, 60), right_bottom=Point(33, 63)),
        ]))

    def test_for_horizontal_returns_two_rectangles(self):
        result = make_tile_barriers(tile_type=TileType.HORIZONTAL,
                                    position=Point(10, 20), margin=1, size=3)
        assert_that(result, equal_to([
            Rectangle(left_top=Point(30, 60), right_bottom=Point(33, 61)),
            Rectangle(left_top=Point(30, 62), right_bottom=Point(33, 63)),
        ]))

    def test_for_left_top_corner_returns_two_rectangles_and_one_circle(self):
        result = make_tile_barriers(tile_type=TileType.LEFT_TOP_CORNER,
                                    position=Point(10, 20), margin=1, size=3)
        assert_that(result, equal_to([
            Rectangle(left_top=Point(30, 60), right_bottom=Point(31, 63)),
            Rectangle(left_top=Point(30, 60), right_bottom=Point(33, 61)),
            Circle(Point(33, 63), 1),
        ]))

    def test_for_crossroads_returns_four_circles(self):
        result = make_tile_barriers(tile_type=TileType.CROSSROADS,
                                    position=Point(10, 20), margin=1, size=3)
        assert_that(result, equal_to([
            Circle(Point(30, 60), 1), Circle(Point(30, 63), 1),
            Circle(Point(33, 60), 1), Circle(Point(33, 63), 1),
        ]))

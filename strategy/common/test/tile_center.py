from unittest import TestCase
from hamcrest import assert_that, equal_to
from strategy.common import Point, get_tile_center
from strategy.common.tile_center import tile_center_coord


class TileCenterCoordTest(TestCase):
    def test_at_0_for_tile_size_10_returns_5(self):
        assert_that(tile_center_coord(value=0, size=10), equal_to(5))

    def test_at_1_for_tile_size_10_returns_15(self):
        assert_that(tile_center_coord(value=1, size=10), equal_to(15))


class TileCenterTest(TestCase):
    def test_at_point_0_0_for_tile_size_10_returns_point_5_5(self):
        assert_that(get_tile_center(point=Point(0, 0), size=10),
                    equal_to(Point(5, 5)))

    def test_at_point_0_1_for_tile_size_10_returns_point_5_15(self):
        assert_that(get_tile_center(point=Point(0, 1), size=10),
                    equal_to(Point(5, 15)))

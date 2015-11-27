from unittest import TestCase
from hamcrest import assert_that, equal_to
from strategy.common import Point
from strategy.common.current_tile import get_current_tile, tile_coord


class TileCoordTest(TestCase):
    def test_at_0_with_tile_size_100_returns_0(self):
        assert_that(tile_coord(value=0, tile_size=100), equal_to(0))

    def test_at_99_with_tile_size_100_returns_0(self):
        assert_that(tile_coord(value=99, tile_size=100), equal_to(0))

    def test_at_100_with_tile_size_100_returns_1(self):
        assert_that(tile_coord(value=100, tile_size=100), equal_to(1))


class CurrentTileTest(TestCase):
    def test_at_100_100_with_tile_size_100_returns_1_1(self):
        result = get_current_tile(point=Point(x=100, y=100), tile_size=100)
        assert_that(result, equal_to(Point(1, 1)))

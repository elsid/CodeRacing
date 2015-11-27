from unittest import TestCase
from hamcrest import assert_that, equal_to
from strategy.common import Point
from strategy.barriers import make_passability_function
from strategy.barriers.circle import Circle


class MakeTilePassabilityTest(TestCase):
    def test_inside_one_of_circles_returns_0(self):
        passability = make_passability_function(
            barriers=[Circle(Point(0, 0), 1), Circle(Point(1, 1), 1)],
            radius=1, speed=Point(0, 0), tiles=[Point(0, 0)], tile_size=4)
        assert_that(passability(0, 0), equal_to(0))

    def test_inside_all_of_circles_returns_1(self):
        passability = make_passability_function(
            barriers=[Circle(Point(0, 0), 1), Circle(Point(1, 1), 1)],
            radius=1, speed=Point(0, 0), tiles=[Point(0, 0)], tile_size=4)
        assert_that(passability(3, 3), equal_to(1))

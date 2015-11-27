from unittest import TestCase
from hamcrest import assert_that, equal_to
from strategy.common import Line, Point


class LineTest(TestCase):
    def test_nearest_at_line_returns_equal(self):
        line = Line(begin=Point(0, 0), end=Point(1, 0))
        result = line.nearest(Point(0, 0))
        assert_that(result, equal_to(Point(0, 0)))

    def test_nearest_at_line_but_not_in_segment_returns_equal(self):
        line = Line(begin=Point(0, 0), end=Point(1, 0))
        result = line.nearest(Point(2, 0))
        assert_that(result, equal_to(Point(2, 0)))

    def test_nearest_not_at_line_returns_at_line(self):
        line = Line(begin=Point(0, 0), end=Point(1, 0))
        result = line.nearest(Point(1, 1))
        assert_that(result, equal_to(Point(1, 0)))

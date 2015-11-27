from unittest import TestCase
from hamcrest import assert_that, equal_to
from strategy.common import Point
from strategy.barriers.rectangle import Rectangle


class RectangleTest(TestCase):
    def test_point_code_inside_returns_inside(self):
        border = Rectangle(left_top=Point(-1, -1), right_bottom=Point(1, 1))
        result = border.point_code(Point(0, 0))
        assert_that(result, equal_to(Rectangle.INSIDE))

    def test_point_code_at_left_returns_left(self):
        border = Rectangle(left_top=Point(-1, -1), right_bottom=Point(1, 1))
        result = border.point_code(Point(-2, 0))
        assert_that(result, equal_to(Rectangle.LEFT))

    def test_point_code_at_right_returns_right(self):
        border = Rectangle(left_top=Point(-1, -1), right_bottom=Point(1, 1))
        result = border.point_code(Point(2, 0))
        assert_that(result, equal_to(Rectangle.RIGHT))

    def test_point_code_at_top_returns_top(self):
        border = Rectangle(left_top=Point(-1, -1), right_bottom=Point(1, 1))
        result = border.point_code(Point(0, -2))
        assert_that(result, equal_to(Rectangle.TOP))

    def test_point_code_at_bottom_returns_bottom(self):
        border = Rectangle(left_top=Point(-1, -1), right_bottom=Point(1, 1))
        result = border.point_code(Point(0, 2))
        assert_that(result, equal_to(Rectangle.BOTTOM))

    def test_point_code_at_left_top_returns_left_top(self):
        border = Rectangle(left_top=Point(-1, -1), right_bottom=Point(1, 1))
        result = border.point_code(Point(-2, -2))
        assert_that(result, equal_to(Rectangle.LEFT | Rectangle.TOP))

    def test_point_code_at_left_bottom_returns_left_bottom(self):
        border = Rectangle(left_top=Point(-1, -1), right_bottom=Point(1, 1))
        result = border.point_code(Point(-2, 2))
        assert_that(result, equal_to(Rectangle.LEFT | Rectangle.BOTTOM))

    def test_point_code_at_right_top_returns_right_top(self):
        border = Rectangle(left_top=Point(-1, -1), right_bottom=Point(1, 1))
        result = border.point_code(Point(2, -2))
        assert_that(result, equal_to(Rectangle.RIGHT | Rectangle.TOP))

    def test_point_code_at_right_bottom_returns_right_bottom(self):
        border = Rectangle(left_top=Point(-1, -1), right_bottom=Point(1, 1))
        result = border.point_code(Point(2, 2))
        assert_that(result, equal_to(Rectangle.RIGHT | Rectangle.BOTTOM))

    def test_passability_for_position_and_border_inside_returns_0(self):
        border = Rectangle(left_top=Point(-1, -1), right_bottom=Point(1, 1))
        result = border.passability(Point(0, 0), radius=1)
        assert_that(result, equal_to(0))

    def test_passability_for_position_inside_returns_0(self):
        border = Rectangle(left_top=Point(-1, -1), right_bottom=Point(1, 1))
        result = border.passability(Point(0, 0), radius=2)
        assert_that(result, equal_to(0))

    def test_passability_for_border_inside_returns_0(self):
        border = Rectangle(left_top=Point(-1, -1), right_bottom=Point(1, 1))
        result = border.passability(Point(-2, 0), radius=1)
        assert_that(result, equal_to(0))

    def test_passability_for_position_and_border_outside_with_disjoint_codes_returns_0(self):
        border = Rectangle(left_top=Point(-1, -1), right_bottom=Point(1, 1))
        result = border.passability(Point(-3, 0), radius=6)
        assert_that(result, equal_to(0))

    def test_passability_for_position_and_border_outside_with_cross_codes_returns_1(self):
        border = Rectangle(left_top=Point(-1, -1), right_bottom=Point(1, 1))
        result = border.passability(Point(-3, 0), radius=1)
        assert_that(result, equal_to(1))

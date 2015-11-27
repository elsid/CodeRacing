from unittest import TestCase
from hamcrest import assert_that, equal_to
from strategy.common import Point
from strategy.path.adjust_path import (
    input_type,
    output_type,
    SideType,
    TypedPoint,
    PointType,
    point_type,
    adjust_path_point,
)


class InputTypeTest(TestCase):
    def test_for_0_0_and_1_0_returns_left(self):
        result = input_type(previous=Point(0, 0), current=Point(1, 0))
        assert_that(result, equal_to(SideType.LEFT))

    def test_for_1_0_and_0_0_returns_right(self):
        result = input_type(previous=Point(1, 0), current=Point(0, 0))
        assert_that(result, equal_to(SideType.RIGHT))

    def test_for_0_0_and_0_1_returns_top(self):
        result = input_type(previous=Point(0, 0), current=Point(0, 1))
        assert_that(result, equal_to(SideType.TOP))

    def test_for_0_1_and_0_0_returns_bottom(self):
        result = input_type(previous=Point(0, 1), current=Point(0, 0))
        assert_that(result, equal_to(SideType.BOTTOM))

    def test_for_0_1_and_1_0_returns_unknown(self):
        result = input_type(previous=Point(0, 1), current=Point(1, 0))
        assert_that(result, equal_to(SideType.UNKNOWN))


class OutputTypeTest(TestCase):
    def test_for_0_0_and_1_0_returns_left(self):
        result = output_type(current=Point(0, 0),
                             following=Point(1, 0))
        assert_that(result, equal_to(SideType.RIGHT))

    def test_for_1_0_and_0_0_returns_right(self):
        result = output_type(current=Point(1, 0),
                             following=Point(0, 0))
        assert_that(result, equal_to(SideType.LEFT))

    def test_for_0_0_and_0_1_returns_top(self):
        result = output_type(current=Point(0, 0),
                             following=Point(0, 1))
        assert_that(result, equal_to(SideType.BOTTOM))

    def test_for_0_1_and_0_0_returns_bottom(self):
        result = output_type(current=Point(0, 1),
                             following=Point(0, 0))
        assert_that(result, equal_to(SideType.TOP))


class PointTypeTest(TestCase):
    def test_for_0_0_and_1_0_and_2_0_returns_left_right(self):
        result = point_type(previous=Point(0, 0), current=Point(1, 0),
                            following=Point(2, 0))
        assert_that(result, equal_to(PointType.LEFT_RIGHT))

    def test_for_0_1_and_1_1_and_1_0_returns_right_top(self):
        result = point_type(previous=Point(0, 1), current=Point(1, 1),
                            following=Point(1, 0))
        assert_that(result, equal_to(PointType.LEFT_TOP))

    def test_for_0_0_and_1_0_and_1_1_returns_left_bottom(self):
        result = point_type(previous=Point(0, 0), current=Point(1, 0),
                            following=Point(1, 1))
        assert_that(result, equal_to(PointType.LEFT_BOTTOM))


class AdjustPathPointTest(TestCase):
    def test_for_2_2_left_right_and_any_left_right_returns_2_2(self):
        result = adjust_path_point(
            current=TypedPoint(Point(2, 2), PointType.LEFT_RIGHT),
            following=TypedPoint(Point(4, 2), PointType.LEFT_RIGHT),
            tile_size=4)
        assert_that(result, equal_to(Point(2, 2)))

    def test_for_2_2_left_top_and_any_following_and_tile_size_4_returns_1_1(self):
        result = adjust_path_point(
            current=TypedPoint(Point(2, 2), PointType.LEFT_TOP),
            following=TypedPoint(Point(2, 4), PointType.BOTTOM_TOP),
            tile_size=4)
        assert_that(result, equal_to(Point(1, 1)))

    def test_for_2_2_left_right_and_any_left_top_following_and_tile_size_4_returns_2_3(self):
        result = adjust_path_point(
            current=TypedPoint(Point(2, 2), PointType.LEFT_RIGHT),
            following=TypedPoint(Point(4, 2), PointType.LEFT_TOP),
            tile_size=4)
        assert_that(result, equal_to(Point(2, 3)))

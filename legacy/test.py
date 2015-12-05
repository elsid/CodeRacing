from unittest import TestCase, main
from hamcrest import assert_that, equal_to
from math import pi, sqrt
from MyStrategy import (
    Point,
    tile_coord,
    get_current_tile,
    Rectangle,
    Circle,
    make_tile_barriers,
    make_passability_function,
    tile_center_coord,
    get_tile_center,
    shift_to_borders,
    is_direct,
    reduce_direct,
    reduce_direct_first_after_me,
    is_diagonal_direct,
    reduce_diagonal_direct,
    polar,
    take_for_spline,
    make_tile_rectangle,
    Line,
    get_point_input_type,
    get_point_output_type,
    get_point_type,
    adjust_path,
    adjust_path_point,
    SideType,
    PointType,
    TypedPoint,
    shift_on_direct,
)


class PolarTest(TestCase):
    def test_for_empty_with_origin_0_0_returns_empty(self):
        result = polar(origin=Point(0, 0), path=[])
        assert_that(list(result), equal_to([]))

    def test_for_one_with_origin_0_0_returns_one_polar(self):
        result = polar(origin=Point(0, 0), path=[Point(1, 1)])
        assert_that(list(result), equal_to([Point(sqrt(2), pi / 4)]))

    def test_for_one_with_origin_1_0_returns_one_shifted_polar(self):
        result = polar(origin=Point(1, 0), path=[Point(1, 1)])
        assert_that(list(result), equal_to([Point(1, pi / 2)]))


class TakeForSplineTest(TestCase):
    def test_for_empty_returns_empty(self):
        result = take_for_spline([])
        assert_that(list(result), equal_to([]))

    def test_for_one_returns_equal(self):
        result = take_for_spline([Point(0, 0)])
        assert_that(list(result), equal_to([Point(0, 0)]))

    def test_for_two_where_second_radius_greater_than_first_returns_equal(self):
        result = take_for_spline([Point(0, 0), Point(1, 1)])
        assert_that(list(result), equal_to([Point(0, 0), Point(1, 1)]))

    def test_for_two_where_second_radius_equals_first_returns_without_second(self):
        result = take_for_spline([Point(1, 0), Point(1, 1)])
        assert_that(list(result), equal_to([Point(1, 0)]))

    def test_for_two_where_second_radius_less_than_first_returns_without_second(self):
        result = take_for_spline([Point(1, 0), Point(0.5, 0.5)])
        assert_that(list(result), equal_to([Point(1, 0)]))


class MakeTileRectangleTest(TestCase):
    def test_at_1_1_with_size_2_returns_0_0_2_2(self):
        result = make_tile_rectangle(position=Point(0, 0,), size=2)
        assert_that(result, equal_to(Rectangle(left_top=Point(0, 0),
                                               right_bottom=Point(2, 2))))


if __name__ == '__main__':
    main()

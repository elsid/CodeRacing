from unittest import TestCase
from hamcrest import assert_that, equal_to
from model.TileType import TileType
from strategy_common import Point
from strategy_path import (
    reduce_diagonal_direct,
    reduce_direct,
    reduce_direct_first_after_me,
    shift_to_borders,
    is_diagonal_direct,
    is_direct,
    make_path,
    AdjacencyMatrix,
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


class IsDiagonalDirectTest(TestCase):
    def test_for_equal_returns_true(self):
        result = is_diagonal_direct(previous=Point(0, 0),
                                    current=Point(0, 0),
                                    following=Point(0, 0))
        assert_that(result, equal_to(True))

    def test_for_previous_equals_minus_following_and_current_is_middle_of_line_between_returns_true(self):
        result = is_diagonal_direct(previous=Point(-1, -1),
                                    current=Point(0, 0),
                                    following=Point(1, 1))
        assert_that(result, equal_to(True))

    def test_for_previous_equals_minus_following_and_current_is_not_point_of_line_between_returns_false(self):
        result = is_diagonal_direct(previous=Point(-1, -1),
                                    current=Point(1, -1),
                                    following=Point(1, 1))
        assert_that(result, equal_to(False))

    def test_for_previous_equals_minus_following_and_current_is_not_middle_of_line_between_returns_false(self):
        result = is_diagonal_direct(previous=Point(-1, -1),
                                    current=Point(0.5, 0.5),
                                    following=Point(1, 1))
        assert_that(result, equal_to(False))


class ReduceDiagonalDirectTest(TestCase):
    def test_for_empty_returns_empty(self):
        result = reduce_diagonal_direct([])
        assert_that(list(result), equal_to([]))

    def test_for_one_returns_equal(self):
        result = reduce_diagonal_direct([Point(0, 0)])
        assert_that(list(result), equal_to([Point(0, 0)]))

    def test_for_two_returns_equal(self):
        result = reduce_diagonal_direct([Point(0, 0), Point(1, 1)])
        assert_that(list(result), equal_to([Point(0, 0), Point(1, 1)]))

    def test_for_three_not_diagonal_direct_returns_equal(self):
        result = reduce_diagonal_direct(
            [Point(0, 0), Point(1, 1), Point(2, 1)])
        assert_that(list(result),
                    equal_to([Point(0, 0), Point(1, 1), Point(2, 1)]))

    def test_for_three_diagonal_direct_returns_without_second(self):
        result = reduce_diagonal_direct([Point(0, 0), Point(1, 1), Point(2, 2)])
        assert_that(list(result), equal_to([Point(0, 0), Point(2, 2)]))


class ReduceDirectFirstAfterMeTest(TestCase):
    def test_for_empty_returns_empty(self):
        result = reduce_direct_first_after_me([])
        assert_that(list(result), equal_to([]))

    def test_for_one_returns_equal(self):
        result = reduce_direct_first_after_me([Point(0, 0)])
        assert_that(list(result), equal_to([Point(0, 0)]))

    def test_for_two_different_returns_equal(self):
        result = reduce_direct_first_after_me([Point(0, 0), Point(1, 1)])
        assert_that(list(result), equal_to([Point(0, 0), Point(1, 1)]))

    def test_for_two_with_equal_y_returns_without_first(self):
        result = reduce_direct_first_after_me([Point(0, 0), Point(1, 0)])
        assert_that(list(result), equal_to([Point(1, 0)]))

    def test_for_three_where_first_two_with_equal_y_returns_without_first(self):
        result = reduce_direct_first_after_me(
            [Point(0, 0), Point(1, 0), Point(1, 1)])
        assert_that(list(result), equal_to([Point(1, 0), Point(1, 1)]))

    def test_for_three_with_equal_y_returns_without_first(self):
        result = reduce_direct_first_after_me(
            [Point(0, 0), Point(1, 0), Point(2, 0)])
        assert_that(list(result), equal_to([Point(1, 0), Point(2, 0)]))


class IsDirectTest(TestCase):
    def test_for_equal_returns_true(self):
        result = is_direct(previous=Point(0, 0),
                           current=Point(0, 0),
                           following=Point(0, 0))
        assert_that(result, equal_to(True))

    def test_for_all_with_equal_x_returns_false(self):
        result = is_direct(previous=Point(0, 0),
                           current=Point(0, 1),
                           following=Point(0, 2))
        assert_that(result, equal_to(True))

    def test_for_all_with_equal_y_returns_false(self):
        result = is_direct(previous=Point(0, 0),
                           current=Point(1, 0),
                           following=Point(2, 0))
        assert_that(result, equal_to(True))

    def test_for_previous_and_current_with_equal_x_returns_false(self):
        result = is_direct(previous=Point(0, 0),
                           current=Point(0, 1),
                           following=Point(2, 2))
        assert_that(result, equal_to(False))

    def test_for_previous_and_current_with_equal_y_returns_false(self):
        result = is_direct(previous=Point(0, 0),
                           current=Point(0, 1),
                           following=Point(2, 2))
        assert_that(result, equal_to(False))

    def test_for_previous_and_following_with_equal_x_returns_false(self):
        result = is_direct(previous=Point(0, 0),
                           current=Point(1, 1),
                           following=Point(0, 2))
        assert_that(result, equal_to(False))

    def test_for_previous_and_following_with_equal_y_returns_false(self):
        result = is_direct(previous=Point(0, 0),
                           current=Point(1, 1),
                           following=Point(2, 0))
        assert_that(result, equal_to(False))

    def test_for_current_and_following_with_equal_x_returns_false(self):
        result = is_direct(previous=Point(0, 0),
                           current=Point(1, 1),
                           following=Point(1, 2))
        assert_that(result, equal_to(False))

    def test_for_current_and_following_with_equal_y_returns_false(self):
        result = is_direct(previous=Point(0, 0),
                           current=Point(1, 1),
                           following=Point(2, 1))
        assert_that(result, equal_to(False))

    def test_for_all_different_returns_false(self):
        result = is_direct(previous=Point(0, 0),
                           current=Point(1, 1),
                           following=Point(2, 2))
        assert_that(result, equal_to(False))


class ReduceDirectTest(TestCase):
    def test_for_empty_returns_empty(self):
        result = reduce_direct([])
        assert_that(list(result), equal_to([]))

    def test_for_one_returns_equal(self):
        result = reduce_direct([Point(0, 0)])
        assert_that(list(result), equal_to([Point(0, 0)]))

    def test_for_two_returns_equal(self):
        result = reduce_direct([Point(0, 0), Point(1, 1)])
        assert_that(list(result), equal_to([Point(0, 0), Point(1, 1)]))

    def test_for_three_not_direct_returns_equal(self):
        result = reduce_direct([Point(0, 0), Point(1, 0), Point(1, 1)])
        assert_that(list(result),
                    equal_to([Point(0, 0), Point(1, 0), Point(1, 1)]))

    def test_for_three_direct_returns_without_second(self):
        result = reduce_direct([Point(0, 0), Point(1, 0), Point(2, 0)])
        assert_that(list(result), equal_to([Point(0, 0), Point(2, 0)]))


class ShiftToBordersTest(TestCase):
    def test_for_empty_returns_empty(self):
        result = shift_to_borders([])
        assert_that(list(result), equal_to([]))

    def test_for_one_returns_equal(self):
        result = shift_to_borders([Point(0, 0)])
        assert_that(list(result), equal_to([Point(0, 0)]))

    def test_for_two_returns_first_result_by_half_distance_to_second(self):
        result = shift_to_borders([Point(0, 0), Point(1, 0)])
        assert_that(list(result), equal_to([Point(0.5, 0), Point(1, 0)]))

    def test_for_turn_returns_first_and_second_result_to_their_following(self):
        result = shift_to_borders([Point(0, 0), Point(1, 0), Point(1, 1)])
        assert_that(list(result),
                    equal_to([Point(0.5, 0), Point(1, 0.5), Point(1, 1)]))


class AdjacencyMatrixTest(TestCase):
    def test_create_from_empty_returns_one(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY],
        ])
        assert_that(matrix.values, equal_to([
            [0]
        ]))

    def test_create_from_empty_vertical_empty_returns_second_connected_to_first_and_third(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY, TileType.VERTICAL, TileType.EMPTY],
        ])
        assert_that(matrix.values, equal_to([
            [0, 0, 0],
            [1, 0, 1],
            [0, 0, 0],
        ]))

    def test_create_from_empty_horizontal_empty_returns_second_connected_to_first_and_third(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY],
            [TileType.HORIZONTAL],
            [TileType.EMPTY],
        ])
        assert_that(matrix.values, equal_to([
            [0, 0, 0],
            [1, 0, 1],
            [0, 0, 0],
        ]))

    def test_index_of_0_0_for_1x1_returns_0(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY],
        ])
        assert_that(matrix.index(0, 0), equal_to(0))

    def test_index_of_0_0_for_2x2_returns_0(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ])
        assert_that(matrix.index(0, 0), equal_to(0))

    def test_index_of_0_1_for_2x2_returns_1(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ])
        assert_that(matrix.index(0, 1), equal_to(1))

    def test_index_of_1_0_for_2x2_returns_1(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ])
        assert_that(matrix.index(1, 0), equal_to(2))

    def test_index_of_1_1_for_2x2_returns_1(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ])
        assert_that(matrix.index(1, 1), equal_to(3))

    def test_x_position_of_0_for_1x1_returns_0(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY],
        ])
        assert_that(matrix.x_position(0), equal_to(0))

    def test_y_position_of_0_for_1x1_returns_0(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY],
        ])
        assert_that(matrix.y_position(0), equal_to(0))

    def test_x_position_of_0_for_2x2_returns_0(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ])
        assert_that(matrix.x_position(0), equal_to(0))

    def test_y_position_of_0_for_2x2_returns_0(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ])
        assert_that(matrix.y_position(0), equal_to(0))

    def test_x_position_of_1_for_2x2_returns_1(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ])
        assert_that(matrix.x_position(1), equal_to(0))

    def test_y_position_of_1_for_2x2_returns_0(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ])
        assert_that(matrix.y_position(1), equal_to(1))

    def test_x_position_of_2_for_2x2_returns_1(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ])
        assert_that(matrix.x_position(2), equal_to(1))

    def test_y_position_of_2_for_2x2_returns_0(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ])
        assert_that(matrix.y_position(2), equal_to(0))

    def test_x_position_of_3_for_2x2_returns_1(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ])
        assert_that(matrix.x_position(3), equal_to(1))

    def test_y_position_of_3_for_2x2_returns_1(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ])
        assert_that(matrix.y_position(3), equal_to(1))


class MakePathTest(TestCase):
    def test_from_vertical_to_next_vertical_returns_first_and_second_point(self):
        path = make_path(
            start_index=1,
            next_waypoint_index=1,
            matrix=AdjacencyMatrix([
                [TileType.EMPTY, TileType.VERTICAL,
                 TileType.VERTICAL, TileType.EMPTY],
            ]),
            waypoints=[[0, 1], [0, 2]])
        assert_that(list(path), equal_to([Point(0, 1), Point(0, 2)]))

    def test_over_three_vertical_returns_three_points(self):
        path = make_path(
            start_index=1,
            next_waypoint_index=1,
            matrix=AdjacencyMatrix([
                [TileType.EMPTY,
                 TileType.VERTICAL, TileType.VERTICAL, TileType.VERTICAL,
                 TileType.EMPTY],
            ]),
            waypoints=[[0, 1], [0, 2], [0, 3]])
        assert_that(list(path),
                    equal_to([Point(0, 1), Point(0, 2), Point(0, 3)]))

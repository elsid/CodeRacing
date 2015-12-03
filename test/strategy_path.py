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
    make_tiles_path,
    shortest_path_with_direction,
    make_graph,
    Node,
    Arc,
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
    def test_for_2_2_left_right_and_any_left_right_returns_equal(self):
        result = adjust_path_point(
            previous=None,
            current=TypedPoint(Point(2, 2), PointType.LEFT_RIGHT),
            following=TypedPoint(Point(4, 2), PointType.LEFT_RIGHT),
            shift=1)
        assert_that(result, equal_to(Point(2, 2)))

    def test_for_2_2_left_top_and_any_following_and_shift_1_returns_1_1(self):
        result = adjust_path_point(
            previous=None,
            current=TypedPoint(Point(2, 2), PointType.LEFT_TOP),
            following=TypedPoint(Point(2, 4), PointType.BOTTOM_TOP),
            shift=1)
        assert_that(result, equal_to(Point(1, 1)))

    def test_for_2_2_left_right_and_any_left_top_following_and_shift_1_returns_2_3(self):
        result = adjust_path_point(
            previous=None,
            current=TypedPoint(Point(2, 2), PointType.LEFT_RIGHT),
            following=TypedPoint(Point(4, 2), PointType.LEFT_TOP),
            shift=1)
        assert_that(result, equal_to(Point(2, 3)))

    def test_for_2_2_left_right_and_any_top_left_previous_and_shift_1_returns_2_3(self):
        result = adjust_path_point(
            previous=TypedPoint(Point(0, 2), PointType.TOP_RIGHT),
            current=TypedPoint(Point(2, 2), PointType.LEFT_RIGHT),
            following=None,
            shift=1)
        assert_that(result, equal_to(Point(2, 2)))


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
        matrix = AdjacencyMatrix(tiles=[
                [TileType.EMPTY],
            ], direction=Point(0, 0), start_tile=Point(0, 0))
        assert_that(matrix.values, equal_to([
            [0]
        ]))

    def test_create_from_empty_vertical_empty_returns_second_connected_to_first_and_third(self):
        matrix = AdjacencyMatrix(tiles=[
            [TileType.EMPTY, TileType.VERTICAL, TileType.EMPTY],
        ], direction=Point(1, 0), start_tile=Point(1, 0))
        assert_that(matrix.values, equal_to([
            [0, 0, 0],
            [3, 0, 3],
            [0, 0, 0],
        ]))

    def test_create_from_empty_horizontal_empty_returns_second_connected_to_first_and_third(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY],
            [TileType.HORIZONTAL],
            [TileType.EMPTY],
        ], direction=Point(0, 1), start_tile=Point(0, 1))
        assert_that(matrix.values, equal_to([
            [0, 0, 0],
            [3, 0, 3],
            [0, 0, 0],
        ]))

    def test_index_of_0_0_for_1x1_returns_0(self):
        matrix = AdjacencyMatrix([
            [TileType.EMPTY],
        ], direction=Point(1, 0), start_tile=Point(0, 0))
        assert_that(matrix.index(0, 0), equal_to(0))

    def test_index_of_0_0_for_2x2_returns_0(self):
        matrix = AdjacencyMatrix(tiles=[
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ], direction=Point(1, 0), start_tile=Point(0, 0))
        assert_that(matrix.index(0, 0), equal_to(0))

    def test_index_of_0_1_for_2x2_returns_1(self):
        matrix = AdjacencyMatrix(tiles=[
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ], direction=Point(1, 0), start_tile=Point(0, 0))
        assert_that(matrix.index(0, 1), equal_to(1))

    def test_index_of_1_0_for_2x2_returns_1(self):
        matrix = AdjacencyMatrix(tiles=[
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ], direction=Point(1, 0), start_tile=Point(0, 0))
        assert_that(matrix.index(1, 0), equal_to(2))

    def test_index_of_1_1_for_2x2_returns_1(self):
        matrix = AdjacencyMatrix(tiles=[
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ], direction=Point(1, 0), start_tile=Point(0, 0))
        assert_that(matrix.index(1, 1), equal_to(3))

    def test_x_position_of_0_for_1x1_returns_0(self):
        matrix = AdjacencyMatrix(tiles=[
            [TileType.EMPTY],
        ], direction=Point(1, 0), start_tile=Point(0, 0))
        assert_that(matrix.x_position(0), equal_to(0))

    def test_y_position_of_0_for_1x1_returns_0(self):
        matrix = AdjacencyMatrix(tiles=[
            [TileType.EMPTY],
        ], direction=Point(1, 0), start_tile=Point(0, 0))
        assert_that(matrix.y_position(0), equal_to(0))

    def test_x_position_of_0_for_2x2_returns_0(self):
        matrix = AdjacencyMatrix(tiles=[
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ], direction=Point(1, 0), start_tile=Point(0, 0))
        assert_that(matrix.x_position(0), equal_to(0))

    def test_y_position_of_0_for_2x2_returns_0(self):
        matrix = AdjacencyMatrix(tiles=[
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ], direction=Point(1, 0), start_tile=Point(0, 0))
        assert_that(matrix.y_position(0), equal_to(0))

    def test_x_position_of_1_for_2x2_returns_1(self):
        matrix = AdjacencyMatrix(tiles=[
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ], direction=Point(1, 0), start_tile=Point(0, 0))
        assert_that(matrix.x_position(1), equal_to(0))

    def test_y_position_of_1_for_2x2_returns_0(self):
        matrix = AdjacencyMatrix(tiles=[
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ], direction=Point(1, 0), start_tile=Point(0, 0))
        assert_that(matrix.y_position(1), equal_to(1))

    def test_x_position_of_2_for_2x2_returns_1(self):
        matrix = AdjacencyMatrix(tiles=[
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ], direction=Point(1, 0), start_tile=Point(0, 0))
        assert_that(matrix.x_position(2), equal_to(1))

    def test_y_position_of_2_for_2x2_returns_0(self):
        matrix = AdjacencyMatrix(tiles=[
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ], direction=Point(1, 0), start_tile=Point(0, 0))
        assert_that(matrix.y_position(2), equal_to(0))

    def test_x_position_of_3_for_2x2_returns_1(self):
        matrix = AdjacencyMatrix(tiles=[
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ], direction=Point(1, 0), start_tile=Point(0, 0))
        assert_that(matrix.x_position(3), equal_to(1))

    def test_y_position_of_3_for_2x2_returns_1(self):
        matrix = AdjacencyMatrix(tiles=[
            [TileType.EMPTY, TileType.EMPTY],
            [TileType.EMPTY, TileType.EMPTY],
        ], direction=Point(1, 0), start_tile=Point(0, 0))
        assert_that(matrix.y_position(3), equal_to(1))


class MakePathTest(TestCase):
    def test_from_vertical_to_next_vertical_returns_first_and_second_point(self):
        path = make_path(
            start_index=1,
            matrix=AdjacencyMatrix(tiles=[
                [TileType.EMPTY, TileType.VERTICAL,
                 TileType.VERTICAL, TileType.EMPTY],
            ], direction=Point(1, 0), start_tile=Point(0, 1)),
            waypoints=[[0, 1], [0, 2]])
        assert_that(list(path), equal_to([Point(0, 1), Point(0, 2)]))

    def test_over_three_vertical_returns_three_points(self):
        path = make_path(
            start_index=1,
            matrix=AdjacencyMatrix(tiles=[
                [TileType.EMPTY,
                 TileType.VERTICAL, TileType.VERTICAL, TileType.VERTICAL,
                 TileType.EMPTY],
            ], direction=Point(1, 0), start_tile=Point(0, 1)),
            waypoints=[[0, 1], [0, 2], [0, 3]])
        assert_that(list(path),
                    equal_to([Point(0, 1), Point(0, 2), Point(0, 3)]))


class MakeTilesPathTest(TestCase):
    def test_from_vertical_to_next_vertical_returns_first_and_second_point(self):
        result = make_tiles_path(
            start_tile=Point(0, 1),
            waypoints=[[0, 1], [0, 2]],
            tiles=[
                [TileType.EMPTY, TileType.VERTICAL,
                 TileType.VERTICAL, TileType.EMPTY],
            ],
            direction=Point(1, 0),
        )
        assert_that(list(result), equal_to([Point(0, 1), Point(0, 2)]))

    def test_over_three_vertical_returns_three_points(self):
        result = make_tiles_path(
            start_tile=Point(0, 1),
            waypoints=[[0, 1], [0, 2], [0, 3]],
            tiles=[
                [TileType.EMPTY,
                 TileType.VERTICAL, TileType.VERTICAL, TileType.VERTICAL,
                 TileType.EMPTY],
            ],
            direction=Point(1, 0),
        )
        assert_that(list(result),
                    equal_to([Point(0, 1), Point(0, 2), Point(0, 3)]))


# class MakeGraphTest(TestCase):
#     def test_for_two_vertical_between_empty(self):
#         result = make_graph(
#             tiles=[
#                 [TileType.EMPTY, TileType.VERTICAL,
#                  TileType.VERTICAL, TileType.EMPTY],
#             ],
#         )
#         assert_that(result, equal_to({
#             0: Node(position=Point(0, 0), arcs=[]),
#             1: Node(position=Point(0, 1), arcs=[Arc(dst=0, weight=1),
#                                                 Arc(dst=2, weight=1)]),
#             2: Node(position=Point(0, 2), arcs=[Arc(dst=1, weight=1),
#                                                 Arc(dst=3, weight=1)]),
#             3: Node(position=Point(0, 3), arcs=[]),
#         }))
#
#     def test_for_two_horizontal_between_empty(self):
#         result = make_graph(
#             tiles=[
#                 [TileType.EMPTY],
#                 [TileType.HORIZONTAL],
#                 [TileType.HORIZONTAL],
#                 [TileType.EMPTY],
#             ],
#         )
#         assert_that(result, equal_to({
#             0: Node(position=Point(0, 0), arcs=[]),
#             1: Node(position=Point(1, 0), arcs=[Arc(dst=0, weight=1),
#                                                 Arc(dst=2, weight=1)]),
#             2: Node(position=Point(2, 0), arcs=[Arc(dst=1, weight=1),
#                                                 Arc(dst=3, weight=1)]),
#             3: Node(position=Point(3, 0), arcs=[]),
#         }))


class ShortestPathWitDirectionTest(TestCase):
    def test_for_graph_with_one_node_without_arcs_returns_empty(self):
        result = shortest_path_with_direction(
            graph={0: Node(position=Point(0, 0), arcs=[])},
            src=0, dst=0, initial_direction=Point(1, 0))
        assert_that(list(result), equal_to([]))

    def test_for_graph_with_one_node_with_arc_returns_empty(self):
        result = shortest_path_with_direction(
            graph={0: Node(position=Point(0, 0), arcs=[Arc(dst=0, weight=1)])},
            src=0, dst=0, initial_direction=Point(1, 0))
        assert_that(list(result), equal_to([]))

    def test_for_graph_with_two_connected_nodes_returns_second(self):
        result = shortest_path_with_direction(graph={
            0: Node(position=Point(0, 0), arcs=[Arc(dst=1, weight=1)]),
            1: Node(position=Point(1, 0), arcs=[]),
        }, src=0, dst=1, initial_direction=Point(1, 0))
        assert_that(list(result), equal_to([1]))

    def test_for_graph_with_two_disconnected_nodes_returns_empty(self):
        result = shortest_path_with_direction(graph={
            0: Node(position=Point(0, 0), arcs=[]),
            1: Node(position=Point(1, 0), arcs=[]),
        }, src=0, dst=1, initial_direction=Point(1, 0))
        assert_that(list(result), equal_to([]))

    def test_for_quadrant_from_left_top_to_right_bottom_with_direction_to_right_returns_path_over_right_top(self):
        result = shortest_path_with_direction(graph={
            0: Node(position=Point(0, 0), arcs=[Arc(dst=1, weight=1),
                                                Arc(dst=2, weight=1)]),
            1: Node(position=Point(0, 1), arcs=[Arc(dst=3, weight=1)]),
            2: Node(position=Point(1, 0), arcs=[Arc(dst=3, weight=1)]),
            3: Node(position=Point(1, 1), arcs=[]),
        }, src=0, dst=3, initial_direction=Point(1, 0))
        assert_that(list(result), equal_to([2, 3]))

    def test_for_quadrant_from_left_top_to_right_bottom_with_direction_to_bottom_returns_path_over_left_bottom(self):
        result = shortest_path_with_direction(graph={
            0: Node(position=Point(0, 0), arcs=[Arc(dst=1, weight=1),
                                                Arc(dst=2, weight=1)]),
            1: Node(position=Point(0, 1), arcs=[Arc(dst=3, weight=1)]),
            2: Node(position=Point(1, 0), arcs=[Arc(dst=3, weight=1)]),
            3: Node(position=Point(1, 1), arcs=[]),
        }, src=0, dst=3, initial_direction=Point(0, 1))
        assert_that(list(result), equal_to([1, 3]))

    def test_for_quadrant_from_left_top_to_right_top_with_direction_to_bottom_returns_path_direct_to_right_top(self):
        result = shortest_path_with_direction(graph={
            0: Node(position=Point(0, 0), arcs=[Arc(dst=1, weight=1),
                                                Arc(dst=2, weight=1)]),
            1: Node(position=Point(0, 1), arcs=[Arc(dst=3, weight=1)]),
            2: Node(position=Point(1, 0), arcs=[]),
            3: Node(position=Point(1, 1), arcs=[Arc(dst=2, weight=1)]),
        }, src=0, dst=2, initial_direction=Point(0, 1))
        assert_that(list(result), equal_to([2]))

    def test_for_graph_with_two_alternatives_with_initial_direction_to_right_bottom_returns_over_bottom(self):
        result = shortest_path_with_direction(graph={
            0: Node(position=Point(0, 1), arcs=[Arc(dst=1, weight=1),
                                                Arc(dst=2, weight=1)]),
            1: Node(position=Point(0, 0), arcs=[Arc(dst=3, weight=1)]),
            2: Node(position=Point(0, 2), arcs=[Arc(dst=4, weight=1)]),
            3: Node(position=Point(1, 0), arcs=[Arc(dst=5, weight=1)]),
            4: Node(position=Point(1, 2), arcs=[Arc(dst=5, weight=1)]),
            5: Node(position=Point(1, 1), arcs=[]),
        }, src=0, dst=5, initial_direction=Point(1, -1))
        assert_that(list(result), equal_to([1, 3, 5]))

    def test_for_graph_with_two_alternatives_with_initial_direction_to_right_top_returns_over_top(self):
        result = shortest_path_with_direction(graph={
            0: Node(position=Point(0, 1), arcs=[Arc(dst=1, weight=1),
                                                Arc(dst=2, weight=1)]),
            1: Node(position=Point(0, 0), arcs=[Arc(dst=3, weight=1)]),
            2: Node(position=Point(0, 2), arcs=[Arc(dst=4, weight=1)]),
            3: Node(position=Point(1, 0), arcs=[Arc(dst=5, weight=1)]),
            4: Node(position=Point(1, 2), arcs=[Arc(dst=5, weight=1)]),
            5: Node(position=Point(1, 1), arcs=[]),
        }, src=0, dst=5, initial_direction=Point(1, 1))
        assert_that(list(result), equal_to([2, 4, 5]))

    def test_for_graph_with_1(self):
        result = shortest_path_with_direction(graph={
            0: Node(position=Point(0, 1), arcs=[Arc(dst=1, weight=0),
                                                Arc(dst=2, weight=0)]),
            1: Node(position=Point(0, 0), arcs=[Arc(dst=3, weight=0)]),
            2: Node(position=Point(0, 2), arcs=[Arc(dst=4, weight=0)]),
            3: Node(position=Point(1, 0), arcs=[Arc(dst=5, weight=0)]),
            4: Node(position=Point(1, 2), arcs=[Arc(dst=6, weight=0)]),
            5: Node(position=Point(1, 1), arcs=[Arc(dst=6, weight=0)]),
            6: Node(position=Point(2, 2), arcs=[]),
        }, src=0, dst=6, initial_direction=Point(1, -1))
        assert_that(list(result), equal_to([1, 3, 5, 6]))

    def test_for_graph_with_2(self):
        result = shortest_path_with_direction(graph={
            0: Node(position=Point(0, 1), arcs=[Arc(dst=1, weight=0.1),
                                                Arc(dst=2, weight=0.1)]),
            1: Node(position=Point(0, 0), arcs=[Arc(dst=3, weight=0.1)]),
            2: Node(position=Point(0, 2), arcs=[Arc(dst=4, weight=0.1)]),
            3: Node(position=Point(1, 0), arcs=[Arc(dst=5, weight=0.1)]),
            4: Node(position=Point(1, 2), arcs=[Arc(dst=6, weight=0.1)]),
            5: Node(position=Point(1, 1), arcs=[Arc(dst=6, weight=0.1)]),
            6: Node(position=Point(2, 2), arcs=[]),
        }, src=0, dst=6, initial_direction=Point(1, 1))
        assert_that(list(result), equal_to([2, 4, 6]))

    def test_example(self):
        result = shortest_path_with_direction(graph={
            0: Node(position=Point(x=0, y=0), arcs=[Arc(dst=16, weight=0.1), Arc(dst=1, weight=0.1)]),
            1: Node(position=Point(x=0, y=1), arcs=[Arc(dst=17, weight=0.1), Arc(dst=0, weight=0.1), Arc(dst=2, weight=0.1)]),
            2: Node(position=Point(x=0, y=2), arcs=[Arc(dst=18, weight=0.1), Arc(dst=1, weight=0.1), Arc(dst=3, weight=0.1)]),
            3: Node(position=Point(x=0, y=3), arcs=[Arc(dst=19, weight=0.1), Arc(dst=2, weight=0.1), Arc(dst=4, weight=0.1)]), 4: Node(position=Point(x=0, y=4), arcs=[Arc(dst=20, weight=0.1), Arc(dst=3, weight=0.1), Arc(dst=5, weight=0.1)]), 5: Node(position=Point(x=0, y=5), arcs=[Arc(dst=21, weight=0.1), Arc(dst=4, weight=0.1), Arc(dst=6, weight=0.1)]), 6: Node(position=Point(x=0, y=6), arcs=[Arc(dst=22, weight=0.1), Arc(dst=5, weight=0.1), Arc(dst=7, weight=0.1)]), 7: Node(position=Point(x=0, y=7), arcs=[Arc(dst=23, weight=0.1), Arc(dst=6, weight=0.1), Arc(dst=8, weight=0.1)]), 8: Node(position=Point(x=0, y=8), arcs=[Arc(dst=24, weight=0.1), Arc(dst=7, weight=0.1), Arc(dst=9, weight=0.1)]), 9: Node(position=Point(x=0, y=9), arcs=[Arc(dst=25, weight=0.1), Arc(dst=8, weight=0.1), Arc(dst=10, weight=0.1)]), 10: Node(position=Point(x=0, y=10), arcs=[Arc(dst=26, weight=0.1), Arc(dst=9, weight=0.1), Arc(dst=11, weight=0.1)]), 11: Node(position=Point(x=0, y=11), arcs=[Arc(dst=27, weight=0.1), Arc(dst=10, weight=0.1), Arc(dst=12, weight=0.1)]), 12: Node(position=Point(x=0, y=12), arcs=[Arc(dst=28, weight=0.1), Arc(dst=11, weight=0.1), Arc(dst=13, weight=0.1)]), 13: Node(position=Point(x=0, y=13), arcs=[Arc(dst=29, weight=0.1), Arc(dst=12, weight=0.1), Arc(dst=14, weight=0.1)]), 14: Node(position=Point(x=0, y=14), arcs=[Arc(dst=30, weight=0.1), Arc(dst=13, weight=0.1), Arc(dst=15, weight=0.1)]), 15: Node(position=Point(x=0, y=15), arcs=[Arc(dst=31, weight=0.1), Arc(dst=14, weight=0.1)]), 16: Node(position=Point(x=1, y=0), arcs=[Arc(dst=0, weight=0.1), Arc(dst=32, weight=0.1)]), 17: Node(position=Point(x=1, y=1), arcs=[Arc(dst=1, weight=0.1), Arc(dst=18, weight=0.1)]), 18: Node(position=Point(x=1, y=2), arcs=[Arc(dst=2, weight=0.1), Arc(dst=17, weight=0.1), Arc(dst=19, weight=0.1)]), 19: Node(position=Point(x=1, y=3), arcs=[Arc(dst=3, weight=0.1), Arc(dst=18, weight=0.1), Arc(dst=20, weight=0.1)]), 20: Node(position=Point(x=1, y=4), arcs=[Arc(dst=4, weight=0.1), Arc(dst=19, weight=0.1), Arc(dst=21, weight=0.1)]), 21: Node(position=Point(x=1, y=5), arcs=[Arc(dst=5, weight=0.1), Arc(dst=20, weight=0.1), Arc(dst=22, weight=0.1)]), 22: Node(position=Point(x=1, y=6), arcs=[Arc(dst=6, weight=0.1), Arc(dst=21, weight=0.1), Arc(dst=23, weight=0.1)]), 23: Node(position=Point(x=1, y=7), arcs=[Arc(dst=7, weight=0.1), Arc(dst=22, weight=0.1), Arc(dst=24, weight=0.1)]), 24: Node(position=Point(x=1, y=8), arcs=[Arc(dst=8, weight=0.1), Arc(dst=23, weight=0.1), Arc(dst=25, weight=0.1)]), 25: Node(position=Point(x=1, y=9), arcs=[Arc(dst=9, weight=0.1), Arc(dst=24, weight=0.1), Arc(dst=26, weight=0.1)]), 26: Node(position=Point(x=1, y=10), arcs=[Arc(dst=10, weight=0.1), Arc(dst=25, weight=0.1), Arc(dst=27, weight=0.1)]), 27: Node(position=Point(x=1, y=11), arcs=[Arc(dst=11, weight=0.1), Arc(dst=26, weight=0.1), Arc(dst=28, weight=0.1)]), 28: Node(position=Point(x=1, y=12), arcs=[Arc(dst=12, weight=0.1), Arc(dst=27, weight=0.1), Arc(dst=29, weight=0.1)]), 29: Node(position=Point(x=1, y=13), arcs=[Arc(dst=13, weight=0.1), Arc(dst=28, weight=0.1), Arc(dst=30, weight=0.1)]), 30: Node(position=Point(x=1, y=14), arcs=[Arc(dst=14, weight=0.1), Arc(dst=29, weight=0.1), Arc(dst=31, weight=0.1)]), 31: Node(position=Point(x=1, y=15), arcs=[Arc(dst=30, weight=0.1), Arc(dst=15, weight=0.1), Arc(dst=47, weight=0.1)]), 32: Node(position=Point(x=2, y=0), arcs=[Arc(dst=16, weight=0.1), Arc(dst=33, weight=0.1)]), 33: Node(position=Point(x=2, y=1), arcs=[Arc(dst=49, weight=0.1), Arc(dst=32, weight=0.1), Arc(dst=34, weight=0.1)]), 34: Node(position=Point(x=2, y=2), arcs=[Arc(dst=50, weight=0.1), Arc(dst=33, weight=0.1), Arc(dst=35, weight=0.1)]), 35: Node(position=Point(x=2, y=3), arcs=[Arc(dst=34, weight=0.1), Arc(dst=36, weight=0.1)]), 36: Node(position=Point(x=2, y=4), arcs=[Arc(dst=35, weight=0.1), Arc(dst=37, weight=0.1)]), 37: Node(position=Point(x=2, y=5), arcs=[Arc(dst=36, weight=0.1), Arc(dst=38, weight=0.1)]), 38: Node(position=Point(x=2, y=6), arcs=[Arc(dst=37, weight=0.1), Arc(dst=39, weight=0.1)]), 39: Node(position=Point(x=2, y=7), arcs=[Arc(dst=38, weight=0.1), Arc(dst=40, weight=0.1)]), 40: Node(position=Point(x=2, y=8), arcs=[Arc(dst=39, weight=0.1), Arc(dst=41, weight=0.1)]), 41: Node(position=Point(x=2, y=9), arcs=[Arc(dst=40, weight=0.1), Arc(dst=42, weight=0.1)]), 42: Node(position=Point(x=2, y=10), arcs=[Arc(dst=41, weight=0.1), Arc(dst=43, weight=0.1)]), 43: Node(position=Point(x=2, y=11), arcs=[Arc(dst=59, weight=0.1), Arc(dst=42, weight=0.1), Arc(dst=44, weight=0.1)]), 44: Node(position=Point(x=2, y=12), arcs=[Arc(dst=60, weight=0.1), Arc(dst=43, weight=0.1), Arc(dst=45, weight=0.1)]), 45: Node(position=Point(x=2, y=13), arcs=[Arc(dst=44, weight=0.1), Arc(dst=46, weight=0.1)]), 46: Node(position=Point(x=2, y=14), arcs=[Arc(dst=62, weight=0.1), Arc(dst=45, weight=0.1)]), 47: Node(position=Point(x=2, y=15), arcs=[Arc(dst=31, weight=0.1), Arc(dst=63, weight=0.1)]), 48: Node(position=Point(x=3, y=0), arcs=[]), 49: Node(position=Point(x=3, y=1), arcs=[Arc(dst=33, weight=0.1), Arc(dst=50, weight=0.1)]), 50: Node(position=Point(x=3, y=2), arcs=[Arc(dst=34, weight=0.1), Arc(dst=49, weight=0.1), Arc(dst=51, weight=0.1)]), 51: Node(position=Point(x=3, y=3), arcs=[Arc(dst=50, weight=0.1), Arc(dst=52, weight=0.1)]), 52: Node(position=Point(x=3, y=4), arcs=[Arc(dst=51, weight=0.1), Arc(dst=53, weight=0.1)]), 53: Node(position=Point(x=3, y=5), arcs=[Arc(dst=52, weight=0.1), Arc(dst=54, weight=0.1)]), 54: Node(position=Point(x=3, y=6), arcs=[Arc(dst=53, weight=0.1), Arc(dst=55, weight=0.1)]), 55: Node(position=Point(x=3, y=7), arcs=[Arc(dst=54, weight=0.1), Arc(dst=56, weight=0.1)]), 56: Node(position=Point(x=3, y=8), arcs=[Arc(dst=55, weight=0.1), Arc(dst=57, weight=0.1)]), 57: Node(position=Point(x=3, y=9), arcs=[Arc(dst=56, weight=0.1), Arc(dst=58, weight=0.1)]), 58: Node(position=Point(x=3, y=10), arcs=[Arc(dst=57, weight=0.1), Arc(dst=59, weight=0.1)]), 59: Node(position=Point(x=3, y=11), arcs=[Arc(dst=43, weight=0.1), Arc(dst=58, weight=0.1), Arc(dst=60, weight=0.1)]), 60: Node(position=Point(x=3, y=12), arcs=[Arc(dst=44, weight=0.1), Arc(dst=59, weight=0.1)]), 61: Node(position=Point(x=3, y=13), arcs=[Arc(dst=77, weight=0.1), Arc(dst=62, weight=0.1)]), 62: Node(position=Point(x=3, y=14), arcs=[Arc(dst=61, weight=0.1), Arc(dst=46, weight=0.1), Arc(dst=78, weight=0.1)]), 63: Node(position=Point(x=3, y=15), arcs=[Arc(dst=47, weight=0.1), Arc(dst=79, weight=0.1)]), 64: Node(position=Point(x=4, y=0), arcs=[]), 65: Node(position=Point(x=4, y=1), arcs=[]), 66: Node(position=Point(x=4, y=2), arcs=[]), 67: Node(position=Point(x=4, y=3), arcs=[]), 68: Node(position=Point(x=4, y=4), arcs=[]), 69: Node(position=Point(x=4, y=5), arcs=[]), 70: Node(position=Point(x=4, y=6), arcs=[]), 71: Node(position=Point(x=4, y=7), arcs=[]), 72: Node(position=Point(x=4, y=8), arcs=[]), 73: Node(position=Point(x=4, y=9), arcs=[]), 74: Node(position=Point(x=4, y=10), arcs=[]), 75: Node(position=Point(x=4, y=11), arcs=[]), 76: Node(position=Point(x=4, y=12), arcs=[]), 77: Node(position=Point(x=4, y=13), arcs=[Arc(dst=78, weight=0.1), Arc(dst=61, weight=0.1), Arc(dst=93, weight=0.1)]), 78: Node(position=Point(x=4, y=14), arcs=[Arc(dst=77, weight=0.1), Arc(dst=62, weight=0.1), Arc(dst=94, weight=0.1)]), 79: Node(position=Point(x=4, y=15), arcs=[Arc(dst=63, weight=0.1), Arc(dst=95, weight=0.1)]), 80: Node(position=Point(x=5, y=0), arcs=[]), 81: Node(position=Point(x=5, y=1), arcs=[]), 82: Node(position=Point(x=5, y=2), arcs=[]), 83: Node(position=Point(x=5, y=3), arcs=[]), 84: Node(position=Point(x=5, y=4), arcs=[]), 85: Node(position=Point(x=5, y=5), arcs=[]), 86: Node(position=Point(x=5, y=6), arcs=[]), 87: Node(position=Point(x=5, y=7), arcs=[]), 88: Node(position=Point(x=5, y=8), arcs=[]), 89: Node(position=Point(x=5, y=9), arcs=[]), 90: Node(position=Point(x=5, y=10), arcs=[]), 91: Node(position=Point(x=5, y=11), arcs=[]), 92: Node(position=Point(x=5, y=12), arcs=[]), 93: Node(position=Point(x=5, y=13), arcs=[Arc(dst=94, weight=0.1), Arc(dst=77, weight=0.1), Arc(dst=109, weight=0.1)]), 94: Node(position=Point(x=5, y=14), arcs=[Arc(dst=93, weight=0.1), Arc(dst=78, weight=0.1), Arc(dst=110, weight=0.1)]), 95: Node(position=Point(x=5, y=15), arcs=[Arc(dst=79, weight=0.1), Arc(dst=111, weight=0.1)]), 96: Node(position=Point(x=6, y=0), arcs=[]), 97: Node(position=Point(x=6, y=1), arcs=[]), 98: Node(position=Point(x=6, y=2), arcs=[]), 99: Node(position=Point(x=6, y=3), arcs=[]), 100: Node(position=Point(x=6, y=4), arcs=[]), 101: Node(position=Point(x=6, y=5), arcs=[]), 102: Node(position=Point(x=6, y=6), arcs=[]), 103: Node(position=Point(x=6, y=7), arcs=[]), 104: Node(position=Point(x=6, y=8), arcs=[]), 105: Node(position=Point(x=6, y=9), arcs=[]), 106: Node(position=Point(x=6, y=10), arcs=[]), 107: Node(position=Point(x=6, y=11), arcs=[]), 108: Node(position=Point(x=6, y=12), arcs=[]),
            109: Node(position=Point(x=6, y=13), arcs=[Arc(dst=110, weight=0.1), Arc(dst=93, weight=0.1), Arc(dst=125, weight=0.1)]),
            110: Node(position=Point(x=6, y=14), arcs=[Arc(dst=94, weight=0.1), Arc(dst=109, weight=0.1)]), 111: Node(position=Point(x=6, y=15), arcs=[Arc(dst=95, weight=0.1), Arc(dst=127, weight=0.1)]), 112: Node(position=Point(x=7, y=0), arcs=[]), 113: Node(position=Point(x=7, y=1), arcs=[]), 114: Node(position=Point(x=7, y=2), arcs=[]), 115: Node(position=Point(x=7, y=3), arcs=[]), 116: Node(position=Point(x=7, y=4), arcs=[]), 117: Node(position=Point(x=7, y=5), arcs=[]), 118: Node(position=Point(x=7, y=6), arcs=[]), 119: Node(position=Point(x=7, y=7), arcs=[]), 120: Node(position=Point(x=7, y=8), arcs=[]), 121: Node(position=Point(x=7, y=9), arcs=[]), 122: Node(position=Point(x=7, y=10), arcs=[]), 123: Node(position=Point(x=7, y=11), arcs=[]),
            124: Node(position=Point(x=7, y=12), arcs=[Arc(dst=140, weight=0.1), Arc(dst=125, weight=0.1)]),
            125: Node(position=Point(x=7, y=13), arcs=[Arc(dst=109, weight=0.1), Arc(dst=124, weight=0.1), Arc(dst=126, weight=0.1)]),
            126: Node(position=Point(x=7, y=14), arcs=[Arc(dst=142, weight=0.1), Arc(dst=125, weight=0.1)]), 127: Node(position=Point(x=7, y=15), arcs=[Arc(dst=111, weight=0.1), Arc(dst=143, weight=0.1)]), 128: Node(position=Point(x=8, y=0), arcs=[]), 129: Node(position=Point(x=8, y=1), arcs=[]), 130: Node(position=Point(x=8, y=2), arcs=[]), 131: Node(position=Point(x=8, y=3), arcs=[]), 132: Node(position=Point(x=8, y=4), arcs=[]), 133: Node(position=Point(x=8, y=5), arcs=[]), 134: Node(position=Point(x=8, y=6), arcs=[]), 135: Node(position=Point(x=8, y=7), arcs=[]), 136: Node(position=Point(x=8, y=8), arcs=[]), 137: Node(position=Point(x=8, y=9), arcs=[]), 138: Node(position=Point(x=8, y=10), arcs=[]), 139: Node(position=Point(x=8, y=11), arcs=[]),
            140: Node(position=Point(x=8, y=12), arcs=[Arc(dst=124, weight=0.1), Arc(dst=156, weight=0.1)]),
            141: Node(position=Point(x=8, y=13), arcs=[]), 142: Node(position=Point(x=8, y=14), arcs=[Arc(dst=126, weight=0.1), Arc(dst=158, weight=0.1)]), 143: Node(position=Point(x=8, y=15), arcs=[Arc(dst=127, weight=0.1), Arc(dst=159, weight=0.1)]), 144: Node(position=Point(x=9, y=0), arcs=[]), 145: Node(position=Point(x=9, y=1), arcs=[]), 146: Node(position=Point(x=9, y=2), arcs=[]), 147: Node(position=Point(x=9, y=3), arcs=[]), 148: Node(position=Point(x=9, y=4), arcs=[]), 149: Node(position=Point(x=9, y=5), arcs=[]), 150: Node(position=Point(x=9, y=6), arcs=[]), 151: Node(position=Point(x=9, y=7), arcs=[]), 152: Node(position=Point(x=9, y=8), arcs=[]), 153: Node(position=Point(x=9, y=9), arcs=[]), 154: Node(position=Point(x=9, y=10), arcs=[]), 155: Node(position=Point(x=9, y=11), arcs=[]), 156: Node(position=Point(x=9, y=12), arcs=[Arc(dst=140, weight=0.1), Arc(dst=157, weight=0.1)]), 157: Node(position=Point(x=9, y=13), arcs=[Arc(dst=173, weight=0.1), Arc(dst=156, weight=0.1), Arc(dst=158, weight=0.1)]), 158: Node(position=Point(x=9, y=14), arcs=[Arc(dst=157, weight=0.1), Arc(dst=142, weight=0.1), Arc(dst=174, weight=0.1)]), 159: Node(position=Point(x=9, y=15), arcs=[Arc(dst=143, weight=0.1), Arc(dst=175, weight=0.1)]), 160: Node(position=Point(x=10, y=0), arcs=[]), 161: Node(position=Point(x=10, y=1), arcs=[]), 162: Node(position=Point(x=10, y=2), arcs=[]), 163: Node(position=Point(x=10, y=3), arcs=[]), 164: Node(position=Point(x=10, y=4), arcs=[]), 165: Node(position=Point(x=10, y=5), arcs=[]), 166: Node(position=Point(x=10, y=6), arcs=[]), 167: Node(position=Point(x=10, y=7), arcs=[]), 168: Node(position=Point(x=10, y=8), arcs=[]), 169: Node(position=Point(x=10, y=9), arcs=[]), 170: Node(position=Point(x=10, y=10), arcs=[]), 171: Node(position=Point(x=10, y=11), arcs=[]), 172: Node(position=Point(x=10, y=12), arcs=[]), 173: Node(position=Point(x=10, y=13), arcs=[Arc(dst=174, weight=0.1), Arc(dst=157, weight=0.1), Arc(dst=189, weight=0.1)]), 174: Node(position=Point(x=10, y=14), arcs=[Arc(dst=173, weight=0.1), Arc(dst=158, weight=0.1), Arc(dst=190, weight=0.1)]), 175: Node(position=Point(x=10, y=15), arcs=[Arc(dst=159, weight=0.1), Arc(dst=191, weight=0.1)]), 176: Node(position=Point(x=11, y=0), arcs=[]), 177: Node(position=Point(x=11, y=1), arcs=[]), 178: Node(position=Point(x=11, y=2), arcs=[]), 179: Node(position=Point(x=11, y=3), arcs=[]), 180: Node(position=Point(x=11, y=4), arcs=[]), 181: Node(position=Point(x=11, y=5), arcs=[]), 182: Node(position=Point(x=11, y=6), arcs=[]), 183: Node(position=Point(x=11, y=7), arcs=[]), 184: Node(position=Point(x=11, y=8), arcs=[]), 185: Node(position=Point(x=11, y=9), arcs=[]), 186: Node(position=Point(x=11, y=10), arcs=[]), 187: Node(position=Point(x=11, y=11), arcs=[]), 188: Node(position=Point(x=11, y=12), arcs=[]), 189: Node(position=Point(x=11, y=13), arcs=[Arc(dst=190, weight=0.1), Arc(dst=173, weight=0.1), Arc(dst=205, weight=0.1)]), 190: Node(position=Point(x=11, y=14), arcs=[Arc(dst=189, weight=0.1), Arc(dst=174, weight=0.1), Arc(dst=206, weight=0.1)]), 191: Node(position=Point(x=11, y=15), arcs=[Arc(dst=175, weight=0.1), Arc(dst=207, weight=0.1)]), 192: Node(position=Point(x=12, y=0), arcs=[]), 193: Node(position=Point(x=12, y=1), arcs=[]), 194: Node(position=Point(x=12, y=2), arcs=[]), 195: Node(position=Point(x=12, y=3), arcs=[]), 196: Node(position=Point(x=12, y=4), arcs=[]), 197: Node(position=Point(x=12, y=5), arcs=[]), 198: Node(position=Point(x=12, y=6), arcs=[]), 199: Node(position=Point(x=12, y=7), arcs=[]), 200: Node(position=Point(x=12, y=8), arcs=[]), 201: Node(position=Point(x=12, y=9), arcs=[]), 202: Node(position=Point(x=12, y=10), arcs=[]), 203: Node(position=Point(x=12, y=11), arcs=[]), 204: Node(position=Point(x=12, y=12), arcs=[]), 205: Node(position=Point(x=12, y=13), arcs=[Arc(dst=206, weight=0.1), Arc(dst=189, weight=0.1), Arc(dst=221, weight=0.1)]), 206: Node(position=Point(x=12, y=14), arcs=[Arc(dst=205, weight=0.1), Arc(dst=190, weight=0.1), Arc(dst=222, weight=0.1)]), 207: Node(position=Point(x=12, y=15), arcs=[Arc(dst=191, weight=0.1), Arc(dst=223, weight=0.1)]), 208: Node(position=Point(x=13, y=0), arcs=[]), 209: Node(position=Point(x=13, y=1), arcs=[]), 210: Node(position=Point(x=13, y=2), arcs=[]), 211: Node(position=Point(x=13, y=3), arcs=[]), 212: Node(position=Point(x=13, y=4), arcs=[]), 213: Node(position=Point(x=13, y=5), arcs=[]), 214: Node(position=Point(x=13, y=6), arcs=[]), 215: Node(position=Point(x=13, y=7), arcs=[]), 216: Node(position=Point(x=13, y=8), arcs=[]), 217: Node(position=Point(x=13, y=9), arcs=[]), 218: Node(position=Point(x=13, y=10), arcs=[]), 219: Node(position=Point(x=13, y=11), arcs=[]), 220: Node(position=Point(x=13, y=12), arcs=[Arc(dst=236, weight=0.1), Arc(dst=221, weight=0.1)]), 221: Node(position=Point(x=13, y=13), arcs=[Arc(dst=205, weight=0.1), Arc(dst=220, weight=0.1), Arc(dst=222, weight=0.1)]), 222: Node(position=Point(x=13, y=14), arcs=[Arc(dst=206, weight=0.1), Arc(dst=221, weight=0.1)]), 223: Node(position=Point(x=13, y=15), arcs=[Arc(dst=207, weight=0.1), Arc(dst=239, weight=0.1)]), 224: Node(position=Point(x=14, y=0), arcs=[]), 225: Node(position=Point(x=14, y=1), arcs=[]), 226: Node(position=Point(x=14, y=2), arcs=[]), 227: Node(position=Point(x=14, y=3), arcs=[]), 228: Node(position=Point(x=14, y=4), arcs=[]), 229: Node(position=Point(x=14, y=5), arcs=[]), 230: Node(position=Point(x=14, y=6), arcs=[]), 231: Node(position=Point(x=14, y=7), arcs=[]), 232: Node(position=Point(x=14, y=8), arcs=[]), 233: Node(position=Point(x=14, y=9), arcs=[]), 234: Node(position=Point(x=14, y=10), arcs=[]), 235: Node(position=Point(x=14, y=11), arcs=[]), 236: Node(position=Point(x=14, y=12), arcs=[Arc(dst=237, weight=0.1), Arc(dst=220, weight=0.1), Arc(dst=252, weight=0.1)]), 237: Node(position=Point(x=14, y=13), arcs=[Arc(dst=236, weight=0.1), Arc(dst=238, weight=0.1)]), 238: Node(position=Point(x=14, y=14), arcs=[Arc(dst=254, weight=0.1), Arc(dst=237, weight=0.1), Arc(dst=239, weight=0.1)]), 239: Node(position=Point(x=14, y=15), arcs=[Arc(dst=223, weight=0.1), Arc(dst=238, weight=0.1)]), 240: Node(position=Point(x=15, y=0), arcs=[]), 241: Node(position=Point(x=15, y=1), arcs=[]), 242: Node(position=Point(x=15, y=2), arcs=[]), 243: Node(position=Point(x=15, y=3), arcs=[]), 244: Node(position=Point(x=15, y=4), arcs=[]), 245: Node(position=Point(x=15, y=5), arcs=[]), 246: Node(position=Point(x=15, y=6), arcs=[]), 247: Node(position=Point(x=15, y=7), arcs=[]), 248: Node(position=Point(x=15, y=8), arcs=[]), 249: Node(position=Point(x=15, y=9), arcs=[]), 250: Node(position=Point(x=15, y=10), arcs=[]), 251: Node(position=Point(x=15, y=11), arcs=[]), 252: Node(position=Point(x=15, y=12), arcs=[Arc(dst=236, weight=0.1), Arc(dst=253, weight=0.1)]), 253: Node(position=Point(x=15, y=13), arcs=[Arc(dst=252, weight=0.1), Arc(dst=254, weight=0.1)]), 254: Node(position=Point(x=15, y=14), arcs=[Arc(dst=238, weight=0.1), Arc(dst=253, weight=0.1)]), 255: Node(position=Point(x=15, y=15), arcs=[])},
            src=125, dst=221,
            initial_direction=Point(x=0.946773485325451, y=-0.32190055527242856),
            # initial_direction=Point(x=1, y=-0.6),
        )
        assert_that(list(result), equal_to([124, 140, 156, 157, 173, 189, 205, 221]))
        # assert_that(list(result), equal_to([126, 142, 158, 174, 190, 206, 222, 221]))

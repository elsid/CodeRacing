from unittest import TestCase
from hamcrest import assert_that, equal_to
from math import sqrt
from model.TileType import TileType
from strategy_common import Point
from strategy_path import (
    reduce_diagonal_direct,
    reduce_direct,
    reduce_direct_first_after_me,
    shift_to_borders,
    is_diagonal_direct,
    is_direct,
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
    split_arcs,
    add_diagonal_arcs,
    shift_on_direct,
    shift_on_direct_x,
    shift_on_direct_y,
    get_index,
    get_point,
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
            shift=1,
            tile_size=4)
        assert_that(result, equal_to(Point(2, 2)))

    def test_for_2_2_left_top_and_any_following_and_shift_1_returns_1_1(self):
        result = adjust_path_point(
            previous=None,
            current=TypedPoint(Point(2, 2), PointType.LEFT_TOP),
            following=TypedPoint(Point(2, 4), PointType.BOTTOM_TOP),
            shift=1,
            tile_size=4,
        )
        assert_that(result, equal_to(Point(1, 1)))

    def test_for_2_2_left_right_and_any_left_top_following_and_shift_1_returns_2_3(self):
        result = adjust_path_point(
            previous=None,
            current=TypedPoint(Point(2, 2), PointType.LEFT_RIGHT),
            following=TypedPoint(Point(4, 2), PointType.LEFT_TOP),
            shift=1,
            tile_size=4,
        )
        assert_that(result, equal_to(Point(2, 3)))

    def test_for_2_2_left_right_and_any_top_left_previous_and_shift_1_returns_2_3(self):
        result = adjust_path_point(
            previous=TypedPoint(Point(0, 2), PointType.TOP_RIGHT),
            current=TypedPoint(Point(2, 2), PointType.LEFT_RIGHT),
            following=None,
            shift=1,
            tile_size=4,
        )
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


class MakeGraphTest(TestCase):
    def test_for_two_vertical_between_empty(self):
        result = make_graph(
            tiles=[
                [TileType.EMPTY, TileType.VERTICAL,
                 TileType.VERTICAL, TileType.EMPTY],
            ],
        )
        assert_that(result, equal_to({
            0: Node(position=Point(0, 0), arcs=[]),
            1: Node(position=Point(0, 1), arcs=[Arc(dst=0, weight=1),
                                                Arc(dst=2, weight=1)]),
            2: Node(position=Point(0, 2), arcs=[Arc(dst=1, weight=1),
                                                Arc(dst=3, weight=1)]),
            3: Node(position=Point(0, 3), arcs=[]),
        }))

    def test_for_two_horizontal_between_empty(self):
        result = make_graph(
            tiles=[
                [TileType.EMPTY],
                [TileType.HORIZONTAL],
                [TileType.HORIZONTAL],
                [TileType.EMPTY],
            ],
        )
        assert_that(result, equal_to({
            0: Node(position=Point(0, 0), arcs=[]),
            1: Node(position=Point(1, 0), arcs=[Arc(dst=0, weight=1),
                                                Arc(dst=2, weight=1)]),
            2: Node(position=Point(2, 0), arcs=[Arc(dst=1, weight=1),
                                                Arc(dst=3, weight=1)]),
            3: Node(position=Point(3, 0), arcs=[]),
        }))


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
            0: Node(position=Point(0, 1), arcs=[Arc(dst=1, weight=1),
                                                Arc(dst=2, weight=1)]),
            1: Node(position=Point(0, 0), arcs=[Arc(dst=3, weight=1)]),
            2: Node(position=Point(0, 2), arcs=[Arc(dst=4, weight=1)]),
            3: Node(position=Point(1, 0), arcs=[Arc(dst=5, weight=1)]),
            4: Node(position=Point(1, 2), arcs=[Arc(dst=6, weight=1)]),
            5: Node(position=Point(1, 1), arcs=[Arc(dst=6, weight=1)]),
            6: Node(position=Point(2, 2), arcs=[]),
        }, src=0, dst=6, initial_direction=Point(1, -1))
        assert_that(list(result), equal_to([1, 3, 5, 6]))

    def test_for_graph_with_2(self):
        result = shortest_path_with_direction(graph={
            0: Node(position=Point(0, 1), arcs=[Arc(dst=1, weight=1),
                                                Arc(dst=2, weight=1)]),
            1: Node(position=Point(0, 0), arcs=[Arc(dst=3, weight=1)]),
            2: Node(position=Point(0, 2), arcs=[Arc(dst=4, weight=1)]),
            3: Node(position=Point(1, 0), arcs=[Arc(dst=5, weight=1)]),
            4: Node(position=Point(1, 2), arcs=[Arc(dst=6, weight=1)]),
            5: Node(position=Point(1, 1), arcs=[Arc(dst=6, weight=1)]),
            6: Node(position=Point(2, 2), arcs=[]),
        }, src=0, dst=6, initial_direction=Point(1, 1))
        assert_that(list(result), equal_to([2, 4, 6]))


class SplitArcsTest(TestCase):
    def test_for_two_double_connected_nodes_returns_with_three_nodes(self):
        result = split_arcs(graph={
            0: Node(position=Point(0, 0), arcs=[Arc(dst=1, weight=1)]),
            1: Node(position=Point(0, 1), arcs=[Arc(dst=0, weight=1)]),
        })
        assert_that(result, equal_to({
            0: Node(position=Point(0, 0), arcs=[Arc(dst=2, weight=0.5)]),
            1: Node(position=Point(0, 1), arcs=[Arc(dst=2, weight=0.5)]),
            2: Node(position=Point(0, 0.5), arcs=[Arc(dst=0, weight=0.5),
                                                  Arc(dst=1, weight=0.5)]),
        }))

    def test_for_quadrant_from_left_top_to_right_bottom_returns_with_new_four_nodes(self):
        result = split_arcs(graph={
            0: Node(position=Point(0, 0), arcs=[Arc(dst=1, weight=1),
                                                Arc(dst=2, weight=1)]),
            1: Node(position=Point(0, 1), arcs=[Arc(dst=3, weight=1)]),
            2: Node(position=Point(1, 0), arcs=[Arc(dst=3, weight=1)]),
            3: Node(position=Point(1, 1), arcs=[]),
        })
        assert_that(result, equal_to({
            0: Node(position=Point(x=0, y=0),
                    arcs=[Arc(dst=4, weight=0.5), Arc(dst=5, weight=0.5)]),
            1: Node(position=Point(x=0, y=1),
                    arcs=[Arc(dst=4, weight=0.5), Arc(dst=6, weight=0.5)]),
            2: Node(position=Point(x=1, y=0),
                    arcs=[Arc(dst=5, weight=0.5), Arc(dst=7, weight=0.5)]),
            3: Node(position=Point(x=1, y=1),
                    arcs=[Arc(dst=6, weight=0.5), Arc(dst=7, weight=0.5)]),
            4: Node(position=Point(x=0.0, y=0.5),
                    arcs=[Arc(dst=0, weight=0.5), Arc(dst=1, weight=0.5)]),
            5: Node(position=Point(x=0.5, y=0.0),
                    arcs=[Arc(dst=0, weight=0.5), Arc(dst=2, weight=0.5)]),
            6: Node(position=Point(x=0.5, y=1.0),
                    arcs=[Arc(dst=1, weight=0.5), Arc(dst=3, weight=0.5)]),
            7: Node(position=Point(x=1.0, y=0.5),
                    arcs=[Arc(dst=2, weight=0.5), Arc(dst=3, weight=0.5)]),
        }))


class AddDiagonalArcsTest(TestCase):
    def test_for_split_quadrant_from_left_top_to_right_bottom_add_eight_new_arcs(self):
        result = add_diagonal_arcs(graph={
            0: Node(position=Point(x=0, y=0),
                    arcs=[Arc(dst=4, weight=0.5), Arc(dst=5, weight=0.5)]),
            1: Node(position=Point(x=0, y=1),
                    arcs=[Arc(dst=4, weight=0.5), Arc(dst=6, weight=0.5)]),
            2: Node(position=Point(x=1, y=0),
                    arcs=[Arc(dst=5, weight=0.5), Arc(dst=7, weight=0.5)]),
            3: Node(position=Point(x=1, y=1),
                    arcs=[Arc(dst=6, weight=0.5), Arc(dst=7, weight=0.5)]),
            4: Node(position=Point(x=0.0, y=0.5),
                    arcs=[Arc(dst=0, weight=0.5), Arc(dst=1, weight=0.5)]),
            5: Node(position=Point(x=0.5, y=0.0),
                    arcs=[Arc(dst=0, weight=0.5), Arc(dst=2, weight=0.5)]),
            6: Node(position=Point(x=0.5, y=1.0),
                    arcs=[Arc(dst=1, weight=0.5), Arc(dst=3, weight=0.5)]),
            7: Node(position=Point(x=1.0, y=0.5),
                    arcs=[Arc(dst=2, weight=0.5), Arc(dst=3, weight=0.5)]),
        })
        assert_that(result, equal_to({
            0: Node(position=Point(x=0, y=0),
                    arcs=[Arc(dst=4, weight=0.5), Arc(dst=5, weight=0.5)]),
            1: Node(position=Point(x=0, y=1),
                    arcs=[Arc(dst=4, weight=0.5), Arc(dst=6, weight=0.5)]),
            2: Node(position=Point(x=1, y=0),
                    arcs=[Arc(dst=5, weight=0.5), Arc(dst=7, weight=0.5)]),
            3: Node(position=Point(x=1, y=1),
                    arcs=[Arc(dst=6, weight=0.5), Arc(dst=7, weight=0.5)]),
            4: Node(position=Point(x=0.0, y=0.5),
                    arcs=[Arc(dst=0, weight=0.5), Arc(dst=1, weight=0.5),
                          Arc(dst=5, weight=sqrt(2) / 2),
                          Arc(dst=6, weight=sqrt(2) / 2)]),
            5: Node(position=Point(x=0.5, y=0.0),
                    arcs=[Arc(dst=0, weight=0.5), Arc(dst=2, weight=0.5),
                          Arc(dst=4, weight=sqrt(2) / 2),
                          Arc(dst=7, weight=sqrt(2) / 2)]),
            6: Node(position=Point(x=0.5, y=1.0),
                    arcs=[Arc(dst=1, weight=0.5), Arc(dst=3, weight=0.5),
                          Arc(dst=4, weight=sqrt(2) / 2),
                          Arc(dst=7, weight=sqrt(2) / 2)]),
            7: Node(position=Point(x=1.0, y=0.5),
                    arcs=[Arc(dst=2, weight=0.5), Arc(dst=3, weight=0.5),
                          Arc(dst=5, weight=sqrt(2) / 2),
                          Arc(dst=6, weight=sqrt(2) / 2)]),
        }))


class ShiftOnDirectXTest(TestCase):
    def test(self):
        last, points = shift_on_direct_x([Point(0, 0), Point(0, 1),
                                          Point(1, 2)])
        assert_that(last, equal_to(2))
        assert_that(list(points), equal_to([Point(1, 0), Point(1, 1)]))


class ShiftOnDirectYTest(TestCase):
    def test(self):
        last, points = shift_on_direct_y([Point(0, 0), Point(1, 0),
                                          Point(2, 1)])
        assert_that(last, equal_to(2))
        assert_that(list(points), equal_to([Point(0, 1), Point(1, 1)]))


class ShiftOnDirectTest(TestCase):
    def test(self):
        result = shift_on_direct([
            Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 1),
            Point(3, 2), Point(3, 3), Point(3, 4), Point(4, 5),
        ])
        assert_that(list(result), equal_to([
            Point(0, 0), Point(1, 1), Point(2, 1), Point(3, 1),
            Point(4, 2), Point(4, 3), Point(4, 4), Point(4, 5),
        ]))


class GetIndexTest(TestCase):
    def test_for_0_0_with_row_size_1_returns_0(self):
        result = get_index(x=0, y=0, row_size=1)
        assert_that(result, equal_to(0))

    def test_for_0_1_with_row_size_2_returns_1(self):
        result = get_index(x=0, y=1, row_size=2)
        assert_that(result, equal_to(1))

    def test_for_1_0_with_row_size_2_returns_2(self):
        result = get_index(x=1, y=0, row_size=2)
        assert_that(result, equal_to(2))

    def test_for_2_3_with_row_size_4_returns_11(self):
        result = get_index(x=2, y=3, row_size=4)
        assert_that(result, equal_to(11))


class GetPointTest(TestCase):
    def test_for_0_with_row_size_1_returns_0_0(self):
        result = get_point(index=0, row_size=1)
        assert_that(result, equal_to(Point(0, 0)))

    def test_for_1_with_row_size_2_returns_0_1(self):
        result = get_point(index=1, row_size=2)
        assert_that(result, equal_to(Point(0, 1)))

    def test_for_2_with_row_size_2_returns_1_0(self):
        result = get_point(index=2, row_size=2)
        assert_that(result, equal_to(Point(1, 0)))

    def test_for_11_with_row_size_4_returns_2_3(self):
        result = get_point(index=11, row_size=4)
        assert_that(result, equal_to(Point(2, 3)))

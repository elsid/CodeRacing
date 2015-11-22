from unittest import TestCase, main
from hamcrest import assert_that, equal_to
from model.TileType import TileType
from math import pi, sqrt
from MyStrategy import (
    AdjacencyMatrix,
    path_to_end,
    Point,
    tile_coord,
    current_tile,
    Rectangle,
    Circle,
    tile_barriers,
    passability_function,
    tile_center_coord,
    tile_center,
    shift_to_borders,
    is_direct,
    reduce_direct,
    reduce_direct_first_after_me,
    is_diagonal_direct,
    reduce_diagonal_direct,
    polar,
    take_for_spline,
)


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


class PointTest(TestCase):
    def test_polar_for_0_0_returns_0_0(self):
        assert_that(Point(0, 0).polar(), equal_to(Point(0, 0)))

    def test_polar_for_1_0_returns_1_0(self):
        assert_that(Point(1, 0).polar(), equal_to(Point(1, 0)))

    def test_polar_for_0_1_returns_1_and_pi_div_2(self):
        assert_that(Point(0, 1).polar(), equal_to(Point(1, pi / 2)))

    def test_polar_for_1_1_returns_sqrt_2_and_pi_div_2(self):
        assert_that(Point(1, 1).polar(), equal_to(Point(sqrt(2), pi / 4)))


class PathToEndTest(TestCase):
    def test_from_vertical_to_next_vertical_returns_first_and_second_point(self):
        path = path_to_end(
            start_index=1,
            next_waypoint_index=1,
            matrix=AdjacencyMatrix([
                [TileType.EMPTY, TileType.VERTICAL,
                 TileType.VERTICAL, TileType.EMPTY],
            ]),
            waypoints=[[0, 1], [0, 2]])
        assert_that(list(path), equal_to([Point(0, 1), Point(0, 2)]))

    def test_over_three_vertical_returns_three_points(self):
        path = path_to_end(
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


class TileCoordTest(TestCase):
    def test_at_0_with_tile_size_100_returns_0(self):
        assert_that(tile_coord(value=0, tile_size=100), equal_to(0))

    def test_at_99_with_tile_size_100_returns_0(self):
        assert_that(tile_coord(value=99, tile_size=100), equal_to(0))

    def test_at_100_with_tile_size_100_returns_1(self):
        assert_that(tile_coord(value=100, tile_size=100), equal_to(1))


class CurrentTileTest(TestCase):
    def test_at_100_100_with_tile_size_100_returns_1_1(self):
        result = current_tile(x=100, y=100, tile_size=100)
        assert_that(result, equal_to(Point(1, 1)))


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


class CircleTest(TestCase):
    def test_passability_outside_radius_with_fit_size_returns_1(self):
        circle = Circle(Point(0, 0), 1)
        result = circle.passability(position=Point(0, 3), radius=1)
        assert_that(result, equal_to(1.0))

    def test_passability_inside_radius_with_fit_size_returns_0(self):
        circle = Circle(Point(0, 0), 1)
        result = circle.passability(position=Point(0, 1), radius=1)
        assert_that(result, equal_to(0.0))

    def test_passability_outside_radius_with_unfit_size_returns_0(self):
        circle = Circle(Point(0, 0), 1)
        result = circle.passability(position=Point(0, 2), radius=2)
        assert_that(result, equal_to(0.0))

    def test_passability_inside_radius_with_unfit_size_returns_0(self):
        circle = Circle(Point(0, 0), 4)
        result = circle.passability(position=Point(0, 1), radius=4)
        assert_that(result, equal_to(0.0))


class TileBarriersTest(TestCase):
    def test_for_empty_returns_empty_list(self):
        result = tile_barriers(tile_type=TileType.EMPTY,
                               position=Point(10, 10), margin=1, size=3)
        assert_that(result, equal_to([]))

    def test_for_vertical_returns_two_rectangles(self):
        result = tile_barriers(tile_type=TileType.VERTICAL,
                               position=Point(10, 20), margin=1, size=3)
        assert_that(result, equal_to([
            Rectangle(left_top=Point(30, 60), right_bottom=Point(31, 63)),
            Rectangle(left_top=Point(32, 60), right_bottom=Point(33, 63)),
        ]))

    def test_for_horizontal_returns_two_rectangles(self):
        result = tile_barriers(tile_type=TileType.HORIZONTAL,
                               position=Point(10, 20), margin=1, size=3)
        assert_that(result, equal_to([
            Rectangle(left_top=Point(30, 60), right_bottom=Point(33, 61)),
            Rectangle(left_top=Point(30, 62), right_bottom=Point(33, 63)),
        ]))

    def test_for_left_top_corner_returns_two_rectangles_and_one_circle(self):
        result = tile_barriers(tile_type=TileType.LEFT_TOP_CORNER,
                               position=Point(10, 20), margin=1, size=3)
        assert_that(result, equal_to([
            Rectangle(left_top=Point(30, 60), right_bottom=Point(31, 63)),
            Rectangle(left_top=Point(30, 60), right_bottom=Point(33, 61)),
            Circle(Point(33, 63), 1),
        ]))

    def test_for_crossroads_returns_four_circles(self):
        result = tile_barriers(tile_type=TileType.CROSSROADS,
                               position=Point(10, 20), margin=1, size=3)
        assert_that(result, equal_to([
            Circle(Point(30, 60), 1), Circle(Point(30, 63), 1),
            Circle(Point(33, 60), 1), Circle(Point(33, 63), 1),
        ]))


class TilePassabilityTest(TestCase):
    def test_inside_one_of_circles_returns_0(self):
        passability = passability_function(barriers=[Circle(Point(0, 0), 1),
                                                     Circle(Point(1, 1), 1)],
                                           radius=1,
                                           speed=Point(0, 0))
        assert_that(passability(0, 0), equal_to(0))

    def test_inside_all_of_circles_returns_1(self):
        passability = passability_function(barriers=[Circle(Point(0, 0), 1),
                                                     Circle(Point(1, 1), 1)],
                                           radius=1,
                                           speed=Point(0, 0))
        assert_that(passability(3, 3), equal_to(1))


class TileCenterCoordTest(TestCase):
    def test_at_0_for_tile_size_10_returns_5(self):
        assert_that(tile_center_coord(value=0, size=10), equal_to(5))

    def test_at_1_for_tile_size_10_returns_15(self):
        assert_that(tile_center_coord(value=1, size=10), equal_to(15))


class TileCenterTest(TestCase):
    def test_at_point_0_0_for_tile_size_10_returns_point_5_5(self):
        assert_that(tile_center(point=Point(0, 0), size=10),
                    equal_to(Point(5, 5)))

    def test_at_point_0_1_for_tile_size_10_returns_point_5_15(self):
        assert_that(tile_center(point=Point(0, 1), size=10),
                    equal_to(Point(5, 15)))


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
        assert_that(list(result),
                    equal_to([Point(0, 0), Point(2, 0)]))


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


if __name__ == '__main__':
    main()

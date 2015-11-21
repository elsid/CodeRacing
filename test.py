from unittest import TestCase, main
from hamcrest import assert_that, equal_to
from model.TileType import TileType
from MyStrategy import (
    AdjacencyMatrix,
    path_to_end,
    Point,
    tile_coord,
    current_tile,
    Border,
    Circle,
    tile_barriers,
    tile_passability,
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


class PathToEndTest(TestCase):
    def test_from_vertical_to_next_vertical_returns_first_and_second_point(self):
        start_index = 1
        next_waypoint_index = 1
        matrix = AdjacencyMatrix([
            [TileType.EMPTY, TileType.VERTICAL,
             TileType.VERTICAL, TileType.EMPTY],
        ])
        waypoints = [[0, 1], [0, 2]]
        path = list(path_to_end(start_index, next_waypoint_index,
                                matrix, waypoints))
        assert_that(path, equal_to([Point(0, 2)]))

    def test_over_three_vertical_returns_three_points(self):
        start_index = 1
        next_waypoint_index = 1
        matrix = AdjacencyMatrix([
            [TileType.EMPTY,
             TileType.VERTICAL, TileType.VERTICAL, TileType.VERTICAL,
             TileType.EMPTY],
        ])
        waypoints = [[0, 1], [0, 2], [0, 3]]
        path = list(path_to_end(start_index, next_waypoint_index,
                                matrix, waypoints))
        assert_that(path, equal_to([Point(0, 2), Point(0, 3)]))


class TileCoordTest(TestCase):
    def test_at_0_with_tile_size_100_returns_0(self):
        assert_that(tile_coord(0, 100), equal_to(0))

    def test_at_99_with_tile_size_100_returns_0(self):
        assert_that(tile_coord(99, 100), equal_to(0))

    def test_at_100_with_tile_size_100_returns_1(self):
        assert_that(tile_coord(100, 100), equal_to(1))


class CurrentTileTest(TestCase):
    def test_at_100_100_with_tile_size_100_returns_1_1(self):
        assert_that(current_tile(100, 100, 100), equal_to(Point(1, 1)))


class BorderTest(TestCase):
    def test_passability_at_normal_side_with_fit_size_returns_1(self):
        border = Border([0, 0], [1, 0], [0, 1])
        assert_that(border.passability(0, 2, 1, 1), equal_to(1.0))

    def test_passability_at_normal_side_with_unfit_size_returns_0(self):
        border = Border([0, 0], [1, 0], [0, 1])
        assert_that(border.passability(0, 1.5, 3, 3), equal_to(0.0))

    def test_passability_at_not_normal_side_with_fit_size_returns_0(self):
        border = Border([0, 0], [1, 0], [0, 1])
        assert_that(border.passability(0, -2, 1, 1), equal_to(0.0))

    def test_passability_at_not_normal_side_with_unfit_size_returns_0(self):
        border = Border([0, 0], [1, 0], [0, 1])
        assert_that(border.passability(0, -2, 3, 3), equal_to(0.0))


class CircleTest(TestCase):
    def test_passability_outside_radius_with_fit_size_returns_1(self):
        circle = Circle([0, 0], 1)
        assert_that(circle.passability(0, 3, 1, 1), equal_to(1.0))

    def test_passability_inside_radius_with_fit_size_returns_0(self):
        circle = Circle([0, 0], 1)
        assert_that(circle.passability(0, 1, 1, 1), equal_to(0.0))

    def test_passability_outside_radius_with_unfit_size_returns_0(self):
        circle = Circle([0, 0], 1)
        assert_that(circle.passability(0, 2, 2, 2), equal_to(0.0))

    def test_passability_inside_radius_with_unfit_size_returns_0(self):
        circle = Circle([0, 0], 4)
        assert_that(circle.passability(0, 1, 4, 4), equal_to(0.0))


class TileBarriersTest(TestCase):
    def test_for_empty_returns_empty_tuple(self):
        barriers = tile_barriers(TileType.EMPTY, 1, 3)
        assert_that(barriers, equal_to(tuple()))

    def test_for_vertical_returns_two_borders(self):
        barriers = tile_barriers(TileType.VERTICAL, 1, 3)
        assert_that(barriers, equal_to((
            Border([1, 0], [1, 3], [1, 0]),
            Border([2, 0], [2, 3], [-1, 0]),
        )))

    def test_for_horizontal_returns_two_borders(self):
        barriers = tile_barriers(TileType.HORIZONTAL, 1, 3)
        assert_that(barriers, equal_to((
            Border([0, 1], [3, 1], [0, 1]),
            Border([0, 2], [3, 2], [0, -1]),
        )))

    def test_for_left_top_corner_returns_two_borders_and_one_circle(self):
        barriers = tile_barriers(TileType.LEFT_TOP_CORNER, 1, 3)
        assert_that(barriers, equal_to((
            Border([1, 0], [1, 3], [1, 0]),
            Border([0, 1], [3, 1], [0, 1]),
            Circle([3, 3], 1),
        )))

    def test_for_crossroads_returns_four_circles(self):
        barriers = tile_barriers(TileType.CROSSROADS, 1, 3)
        assert_that(barriers, equal_to((
            Circle([0, 0], 1), Circle([0, 3], 1),
            Circle([3, 0], 1), Circle([3, 3], 1),
        )))


class TilePassabilityTest(TestCase):
    def test_inside_one_of_circles_returns_0(self):
        passability = tile_passability(barriers=[Circle([0, 0], 1),
                                                 Circle([1, 1], 1)],
                                       height=1, width=1)
        assert_that(passability(0, 0), equal_to(0))

    def test_inside_all_of_circles_returns_1(self):
        passability = tile_passability(barriers=[Circle([0, 0], 1),
                                                 Circle([1, 1], 1)],
                                       height=1, width=1)
        assert_that(passability(3, 3), equal_to(1))


if __name__ == '__main__':
    main()

from unittest import TestCase
from hamcrest import assert_that, equal_to, close_to
from math import sqrt, pi
from strategy_common import (
    Line,
    Point,
    get_tile_center,
    get_current_tile,
    tile_coord,
    tile_center_coord,
    normalize_angle,
)


class TileCoordTest(TestCase):
    def test_at_0_with_tile_size_100_returns_0(self):
        assert_that(tile_coord(value=0, tile_size=100), equal_to(0))

    def test_at_99_with_tile_size_100_returns_0(self):
        assert_that(tile_coord(value=99, tile_size=100), equal_to(0))

    def test_at_100_with_tile_size_100_returns_1(self):
        assert_that(tile_coord(value=100, tile_size=100), equal_to(1))


class CurrentTileTest(TestCase):
    def test_at_100_100_with_tile_size_100_returns_1_1(self):
        result = get_current_tile(point=Point(x=100, y=100), tile_size=100)
        assert_that(result, equal_to(Point(1, 1)))


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


class PointTest(TestCase):
    def test_polar_0_0_returns_0_0(self):
        assert_that(Point(0, 0).polar(), equal_to(Point(0, 0)))

    def test_polar_1_0_returns_1_0(self):
        assert_that(Point(1, 0).polar(), equal_to(Point(1, 0)))

    def test_polar_0_1_returns_1_and_half_pi(self):
        assert_that(Point(0, 1).polar(), equal_to(Point(1, pi / 2)))

    def test_polar_1_1_returns_sqrt_2_and_half_pi(self):
        assert_that(Point(1, 1).polar(), equal_to(Point(sqrt(2), pi / 4)))

    def test_cos_1_0_to_1_0_returns_1(self):
        assert_that(Point(1, 0).cos(Point(1, 0)), equal_to(1))

    def test_cos_1_0_to_0_1_returns_0(self):
        assert_that(Point(1, 0).cos(Point(0, 1)), equal_to(0))

    def test_cos_1_1_to_1_0_returns_sqrt_2_div_2(self):
        assert_that(Point(1, 1).cos(Point(1, 0)),
                    close_to(value=sqrt(2) / 2, delta=1e-8))

    def test_cos_for_1_1_to_0_1_returns_sqrt_2_div_2(self):
        assert_that(Point(1, 1).cos(Point(0, 1)),
                    close_to(value=sqrt(2) / 2, delta=1e-8))

    def test_rotation_1_0_to_equal_returns_0(self):
        assert_that(Point(1, 0).rotation(Point(1, 0)), equal_to(0))

    def test_rotate_1_0_by_0_returns_equal(self):
        result = Point(1, 0).rotate(0)
        assert_that(result, equal_to(Point(1, 0)))

    def test_rotation_1_0_to_0_1_returns_half_pi(self):
        assert_that(Point(1, 0).rotation(Point(0, 1)), equal_to(pi / 2))

    def test_rotate_1_0_by_half_pi_returns_0_1(self):
        result = Point(1, 0).rotate(pi / 2)
        assert_that(result.x, close_to(value=0, delta=1e-8))
        assert_that(result.y, close_to(value=1, delta=1e-8))

    def test_rotation_1_1_to_1_0_returns_minus_quarter_pi(self):
        assert_that(Point(1, 1).rotation(Point(1, 0)),
                    close_to(value=-pi / 4, delta=1e-8))

    def test_rotate_1_1_by_minus_quarter_pi_returns_sqrt_2_0(self):
        result = Point(1, 1).rotate(-pi / 4)
        assert_that(result.x, close_to(value=sqrt(2), delta=1e-8))
        assert_that(result.y, close_to(value=0, delta=1e-8))

    def test_rotation_1_1_to_0_1_returns_quarter_pi(self):
        assert_that(Point(1, 1).rotation(Point(0, 1)),
                    close_to(value=pi / 4, delta=1e-8))

    def test_rotate_1_1_by_quarter_pi_returns_0_sqrt_2(self):
        result = Point(1, 1).rotate(pi / 4)
        assert_that(result.x, close_to(value=0, delta=1e-8))
        assert_that(result.y, close_to(value=sqrt(2), delta=1e-8))

    def test_rotation_1_0_to_minus_1_0_returns_pi(self):
        assert_that(Point(1, 0).rotation(Point(-1, 0)),
                    close_to(value=pi, delta=1e-8))

    def test_rotate_1_0_by_pi_returns_minus_1_0(self):
        result = Point(1, 0).rotate(pi)
        assert_that(result.x, close_to(value=-1, delta=1e-8))
        assert_that(result.y, close_to(value=0, delta=1e-8))

    def test_rotation_minus_1_0_to_1_0_returns_minus_pi(self):
        assert_that(Point(-1, 0).rotation(Point(1, 0)),
                    close_to(value=-pi, delta=1e-8))

    def test_rotate_minus_1_0_by_minus_pi_returns_1_0(self):
        result = Point(-1, 0).rotate(-pi)
        assert_that(result.x, close_to(value=1, delta=1e-8))
        assert_that(result.y, close_to(value=0, delta=1e-8))

    def test_rotation_1_0_to_minus_1_1_returns_three_quarter_pi(self):
        assert_that(Point(1, 0).rotation(Point(-1, 1)),
                    close_to(value=3 * pi / 4, delta=1e-8))

    def test_rotate_1_0_by_three_quarter_pi_returns_minus_half_sqrt_2_half_sqrt_2(self):
        result = Point(1, 0).rotate(3 * pi / 4)
        assert_that(result.x, close_to(value=-sqrt(2) / 2, delta=1e-8))
        assert_that(result.y, close_to(value=sqrt(2) / 2, delta=1e-8))

    def test_rotation_minus_1_1_to_1_0_returns_3_quarter_pi(self):
        assert_that(Point(-1, 1).rotation(Point(1, 0)),
                    close_to(value=-3 * pi / 4, delta=1e-8))

    def test_rotate_minus_1_1_by_minus_three_quarter_pi_returns_sqrt_2_0(self):
        result = Point(-1, 1).rotate(-3 * pi / 4)
        assert_that(result.x, close_to(value=sqrt(2), delta=1e-8))
        assert_that(result.y, close_to(value=0, delta=1e-8))

    def test_projection_2_2_to_1_0_returns_1_0(self):
        result = Point(2, 2).projection(Point(1, 0))
        assert_that(result, equal_to(Point(2, 0)))

    def test_projection_0_0_to_1_0_returns_0_0(self):
        result = Point(0, 0).projection(Point(1, 0))
        assert_that(result, equal_to(Point(0, 0)))

    def test_projection_1_0_to_1_1_returns_sqrt_2_div_2_sqrt_2_div_2(self):
        result = Point(1, 0).projection(Point(1, 1))
        assert_that(result.x, close_to(value=sqrt(2) / 2, delta=1e-8))
        assert_that(result.y, close_to(value=sqrt(2) / 2, delta=1e-8))


class TileCenterCoordTest(TestCase):
    def test_at_0_for_tile_size_10_returns_5(self):
        assert_that(tile_center_coord(value=0, size=10), equal_to(5))

    def test_at_1_for_tile_size_10_returns_15(self):
        assert_that(tile_center_coord(value=1, size=10), equal_to(15))


class TileCenterTest(TestCase):
    def test_at_point_0_0_for_tile_size_10_returns_point_5_5(self):
        assert_that(get_tile_center(point=Point(0, 0), size=10),
                    equal_to(Point(5, 5)))

    def test_at_point_0_1_for_tile_size_10_returns_point_5_15(self):
        assert_that(get_tile_center(point=Point(0, 1), size=10),
                    equal_to(Point(5, 15)))


class TestNormalizeAnlge(TestCase):
    def test_for_greater_than_or_equal_minus_pi_and_less_than_or_equal_pi_returns_equal(self):
        result = normalize_angle(0.3 * pi)
        assert_that(result, equal_to(0.3 * pi))

    def test_for_pi_returns_equal(self):
        result = normalize_angle(pi)
        assert_that(result, equal_to(pi))

    def test_for_minus_pi_returns_equal(self):
        result = normalize_angle(-pi)
        assert_that(result, equal_to(-pi))

    def test_for_2_pi_returns_0(self):
        result = normalize_angle(2 * pi)
        assert_that(result, equal_to(0))

    def test_for_3_pi_returns_minus_pi(self):
        result = normalize_angle(3 * pi)
        assert_that(result, equal_to(-pi))

    def test_for_minus_3_pi_returns_pi(self):
        result = normalize_angle(-3 * pi)
        assert_that(result, equal_to(pi))

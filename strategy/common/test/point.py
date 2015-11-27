from unittest import TestCase
from hamcrest import assert_that, equal_to, close_to
from math import sqrt, pi
from strategy.common import Point


class PointTest(TestCase):
    def test_polar_0_0_returns_0_0(self):
        assert_that(Point(0, 0).polar(), equal_to(Point(0, 0)))

    def test_polar_1_0_returns_1_0(self):
        assert_that(Point(1, 0).polar(), equal_to(Point(1, 0)))

    def test_polar_0_1_returns_1_and_pi_div_2(self):
        assert_that(Point(0, 1).polar(), equal_to(Point(1, pi / 2)))

    def test_polar_1_1_returns_sqrt_2_and_pi_div_2(self):
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

    def test_rotation_1_0_to_1_0_returns_0(self):
        assert_that(Point(1, 0).rotation(Point(1, 0)), equal_to(0))

    def test_rotation_1_0_to_0_1_returns_pi_div_2(self):
        assert_that(Point(1, 0).rotation(Point(0, 1)), equal_to(-pi / 2))

    def test_rotation_1_1_to_1_0_returns_1_div_2(self):
        assert_that(Point(1, 1).rotation(Point(1, 0)),
                    close_to(value=pi / 4, delta=1e-8))

    def test_rotation_1_1_to_0_1_returns_1_div_2(self):
        assert_that(Point(1, 1).rotation(Point(0, 1)),
                    close_to(value=-pi / 4, delta=1e-8))

    # def test_rotation_1_0_to_minus_1_0_returns_pi(self):
    #     assert_that(Point(1, 0).rotation(Point(-1, 0)),
    #                 close_to(value=pi, delta=1e-8))
    #
    # def test_rotation_minus_1_0_to_1_0_returns_pi(self):
    #     assert_that(Point(-1, 0).rotation(Point(1, 0)),
    #                 close_to(value=-pi, delta=1e-8))

    def test_rotation_1_0_to_minus_1_1_returns_minus_3_pi_div_4(self):
        assert_that(Point(1, 0).rotation(Point(-1, 1)),
                    close_to(value=-3 * pi / 4, delta=1e-8))

    def test_rotation_minus_1_1_to_1_0_returns_3_pi_div_4(self):
        assert_that(Point(-1, 1).rotation(Point(1, 0)),
                    close_to(value=3 * pi / 4, delta=1e-8))

    def test_rotate_0_1_by_minus_pi_div_2_returns_1_0(self):
        result = Point(0, 1).rotate(-pi / 2)
        assert_that(result.x, close_to(value=1, delta=1e-8))
        assert_that(result.y, close_to(value=0, delta=1e-8))

    def test_rotate_1_0_by_pi_div_2_returns_0_1(self):
        result = Point(1, 0).rotate(pi / 2)
        assert_that(result.x, close_to(value=0, delta=1e-8))
        assert_that(result.y, close_to(value=1, delta=1e-8))

    def test_rotate_1_0_by_pi_returns_minus_1_0(self):
        result = Point(1, 0).rotate(pi)
        assert_that(result.x, close_to(value=-1, delta=1e-8))
        assert_that(result.y, close_to(value=0, delta=1e-8))

    def test_rotate_minus_1_0_by_minus_pi_returns_1_0(self):
        result = Point(-1, 0).rotate(-pi)
        assert_that(result.x, close_to(value=1, delta=1e-8))
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

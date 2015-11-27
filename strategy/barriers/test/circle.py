from unittest import TestCase
from hamcrest import assert_that, equal_to
from strategy.common import Point, Line
from strategy.barriers.circle import Circle


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

    def test_intersection_with_line_begins_from_circle_position_returns_one_point(self):
        circle = Circle(Point(0, 0), 1)
        line = Line(begin=Point(0, 0), end=Point(2, 0))
        result = circle.intersection_with_line(line)
        assert_that(result, equal_to([Point(1, 0)]))

    def test_intersection_with_line_cross_circle_position_returns_two_points(self):
        circle = Circle(Point(0, 0), 1)
        line = Line(begin=Point(-2, 0), end=Point(2, 0))
        result = circle.intersection_with_line(line)
        assert_that(result, equal_to([Point(-1, 0), Point(1, 0)]))

    def test_intersection_with_line_inside_circle_returns_empty(self):
        circle = Circle(Point(0, 0), 2)
        line = Line(begin=Point(-1, 0), end=Point(1, 0))
        result = circle.intersection_with_line(line)
        assert_that(result, equal_to([]))

    def test_intersection_with_circle_chord_returns_two_points(self):
        circle = Circle(Point(0, 0), 2)
        line = Line(begin=Point(-2, 1), end=Point(2, 1))
        result = circle.intersection_with_line(line)
        assert_that(len(result), equal_to(2))

    def test_intersection_with_circle_tangent_returns_one_points(self):
        circle = Circle(Point(0, 0), 1)
        line = Line(begin=Point(-1, 1), end=Point(1, 1))
        result = circle.intersection_with_line(line)
        assert_that(result, equal_to([Point(0, 1)]))

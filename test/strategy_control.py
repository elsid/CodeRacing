from unittest import TestCase
from hamcrest import assert_that, equal_to
from strategy_common import Point
from strategy_control import DirectionDetector


class DirectionDetectorTest(TestCase):
    def test_for_initial_points_get_returns_direction(self):
        get_direction = DirectionDetector(
            begin=Point(0, 0),
            end=Point(1, 0),
            min_distance=1,
        )
        assert_that(get_direction(), equal_to(Point(1, 0)))

    def test_update_by_nearest_than_min_distance_returns_initial(self):
        get_direction = DirectionDetector(
            begin=Point(0, 0),
            end=Point(1, 0),
            min_distance=1,
        )
        get_direction.update(Point(1, 0.5))
        assert_that(get_direction(), equal_to(Point(1, 0)))

    def test_update_by_far_than_min_distance_returns_updated(self):
        get_direction = DirectionDetector(
            begin=Point(0, 0),
            end=Point(1, 0),
            min_distance=1,
        )
        get_direction.update(Point(0, 2))
        assert_that(get_direction(), equal_to(Point(-1, 2)))

    def test_update_by_exact_min_distance_returns_updated(self):
        get_direction = DirectionDetector(
            begin=Point(0, 0),
            end=Point(1, 0),
            min_distance=1,
        )
        get_direction.update(Point(1, 1))
        assert_that(get_direction(), equal_to(Point(0, 1)))

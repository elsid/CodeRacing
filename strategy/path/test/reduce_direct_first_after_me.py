from unittest import TestCase
from hamcrest import assert_that, equal_to
from strategy.common import Point
from strategy.path import reduce_direct_first_after_me


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

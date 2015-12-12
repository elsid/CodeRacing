from unittest import TestCase
from hamcrest import assert_that, equal_to, greater_than, less_than
from strategy_common import Point
from strategy_control import DirectionDetector, Controller


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


class ControllerTest(TestCase):
    def test_start_direct_forward(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(1, 0),
            angle=0,
            direct_speed=Point(0, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(1, 0),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(1))
        assert_that(result.wheel_turn, equal_to(0))
        assert_that(result.brake, equal_to(False))

    def test_start_direct_backward(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(-1, 0),
            angle=0,
            direct_speed=Point(0, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(-1, 0),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(-1))
        assert_that(result.wheel_turn, equal_to(0))
        assert_that(result.brake, equal_to(False))

    def test_start_direct_forward_with_initial_speed(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(1, 0),
            angle=0,
            direct_speed=Point(1, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(1, 0),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(1))
        assert_that(result.wheel_turn, equal_to(0))
        assert_that(result.brake, equal_to(False))

    def test_start_direct_backward_with_initial_speed(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(-1, 0),
            angle=0,
            direct_speed=Point(-1, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(-1, 0),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(-1))
        assert_that(result.wheel_turn, equal_to(0))
        assert_that(result.brake, equal_to(False))

    def test_start_direct_forward_with_initial_opposite_speed(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(1, 0),
            angle=0,
            direct_speed=Point(-1, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(1, 0),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(1))
        assert_that(result.wheel_turn, equal_to(0))
        assert_that(result.brake, equal_to(True))

    def test_start_direct_backward_with_initial_opposite_speed(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(-1, 0),
            angle=0,
            direct_speed=Point(1, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(-1, 0),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(-1))
        assert_that(result.wheel_turn, equal_to(0))
        assert_that(result.brake, equal_to(True))

    def test_start_direct_forward_turn_left(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(1, -0.1),
            angle=0,
            direct_speed=Point(0, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(1, 0.1),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(1))
        assert_that(result.wheel_turn, less_than(0))
        assert_that(result.brake, equal_to(False))

    def test_start_direct_forward_turn_right(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(1, 0.1),
            angle=0,
            direct_speed=Point(0, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(1, 0.1),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(1))
        assert_that(result.wheel_turn, greater_than(0))
        assert_that(result.brake, equal_to(False))

    def test_start_direct_backward_turn_left(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(-1, -0.1),
            angle=0,
            direct_speed=Point(0, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(-1, -0.1),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(-1))
        assert_that(result.wheel_turn, less_than(0))
        assert_that(result.brake, equal_to(False))

    def test_start_direct_backward_turn_right(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(-1, 0.1),
            angle=0,
            direct_speed=Point(0, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(-1, 0.1),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(-1))
        assert_that(result.wheel_turn, greater_than(0))
        assert_that(result.brake, equal_to(False))

    def test_start_direct_forward_with_initial_speed_turn_left(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(1, -0.1),
            angle=0,
            direct_speed=Point(1, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(1, 0.1),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(1))
        assert_that(result.wheel_turn, less_than(0))
        assert_that(result.brake, equal_to(False))

    def test_start_direct_forward_with_initial_speed_turn_right(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(1, 0.1),
            angle=0,
            direct_speed=Point(1, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(1, 0.1),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(1))
        assert_that(result.wheel_turn, greater_than(0))
        assert_that(result.brake, equal_to(False))

    def test_start_direct_backward_with_initial_speed_turn_left(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(-1, -0.1),
            angle=0,
            direct_speed=Point(-1, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(-1, -0.1),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(-1))
        assert_that(result.wheel_turn, less_than(0))
        assert_that(result.brake, equal_to(False))

    def test_start_direct_backward_with_initial_speed_turn_right(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(-1, 0.1),
            angle=0,
            direct_speed=Point(-1, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(-1, 0.1),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(-1))
        assert_that(result.wheel_turn, greater_than(0))
        assert_that(result.brake, equal_to(False))

    def test_start_direct_forward_with_initial_opposite_speed_turn_left(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(1, -0.1),
            angle=0,
            direct_speed=Point(-1, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(1, 0.1),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(1))
        assert_that(result.wheel_turn, less_than(0))
        assert_that(result.brake, equal_to(True))

    def test_start_direct_forward_with_initial_opposite_speed_turn_right(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(1, 0.1),
            angle=0,
            direct_speed=Point(-1, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(1, 0.1),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(1))
        assert_that(result.wheel_turn, greater_than(0))
        assert_that(result.brake, equal_to(True))

    def test_start_direct_backward_with_initial_opposite_speed_turn_left(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(-1, -0.1),
            angle=0,
            direct_speed=Point(1, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(-1, -0.1),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(-1))
        assert_that(result.wheel_turn, less_than(0))
        assert_that(result.brake, equal_to(True))

    def test_start_direct_backward_with_initial_opposite_speed_turn_right(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(-1, 0.1),
            angle=0,
            direct_speed=Point(1, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(-1, 0.1),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(-1))
        assert_that(result.wheel_turn, greater_than(0))
        assert_that(result.brake, equal_to(True))

    def test_start_left(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(0, -1),
            angle=0,
            direct_speed=Point(0, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(0, -1),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(1))
        assert_that(result.wheel_turn, less_than(0))
        assert_that(result.brake, equal_to(False))

    def test_start_right(self):
        controller = Controller(distance_to_wheels=1)
        result = controller(
            course=Point(0, 1),
            angle=0,
            direct_speed=Point(0, 0),
            angular_speed_angle=0,
            engine_power=0,
            wheel_turn=0,
            target_speed=Point(0, 1),
            tick=0,
        )
        assert_that(result.engine_power, equal_to(1))
        assert_that(result.wheel_turn, greater_than(0))
        assert_that(result.brake, equal_to(False))

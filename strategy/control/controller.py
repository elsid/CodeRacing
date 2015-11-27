from collections import namedtuple
from math import pi
from strategy.common import Point
from strategy.control.pid_controller import PidController

Control = namedtuple('Control', ('engine_power_derivative',
                                 'wheel_turn_derivative',
                                 'brake'))


class Controller:
    def __init__(self, distance_to_wheels, max_engine_power_derivative,
                 angular_speed_factor):
        self.distance_to_wheels = distance_to_wheels
        self.max_engine_power_derivative = max_engine_power_derivative
        self.angular_speed_factor = angular_speed_factor
        self.__engine_power = PidController(1.0, 0.1, 0.7)
        self.__wheel_turn = PidController(0.3, 0.1, 0.1)
        self.__previous_full_speed = Point(0, 0)
        self.__previous_brake = False

    def __call__(self, direction, angle_error, engine_power, wheel_turn,
                 speed, angular_speed, speed_at_target):
        target_angular_speed = angle_error
        angular_speed_error = target_angular_speed - angular_speed
        target_wheel_turn = angular_speed_error
        wheel_turn_error = target_wheel_turn - wheel_turn
        wheel_turn_derivative = self.__wheel_turn(wheel_turn_error)
        target_speed = speed_at_target
        radius = -(direction * self.distance_to_wheels).rotate(pi / 2)
        angular_speed_vec = Point(-radius.y, radius.x) * angular_speed
        full_speed = speed + angular_speed_vec
        self.__previous_full_speed = full_speed
        target_full_speed = target_speed
        full_speed_error = target_full_speed.norm() - full_speed.norm()
        acceleration = (full_speed - self.__previous_full_speed).norm()
        target_acceleration = full_speed_error
        acceleration_error = target_acceleration - acceleration
        target_engine_power = acceleration_error
        engine_power_error = target_engine_power - engine_power
        engine_power_derivative = self.__engine_power(engine_power_error)
        brake = (engine_power_derivative < -self.max_engine_power_derivative and
                 not self.__previous_brake)
        self.__previous_brake = brake
        return Control(engine_power_derivative, wheel_turn_derivative, brake)

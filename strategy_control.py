from collections import namedtuple
from functools import reduce
from itertools import islice
from math import pi, exp
from operator import mul
from strategy_common import Point

Control = namedtuple('Control', ('engine_power_derivative',
                                 'wheel_turn_derivative',
                                 'brake'))


class Controller:
    def __init__(self, distance_to_wheels, max_engine_power_derivative,
                 angular_speed_factor, is_debug):
        self.distance_to_wheels = distance_to_wheels
        self.max_engine_power_derivative = max_engine_power_derivative
        self.angular_speed_factor = angular_speed_factor
        self.__engine_power = PidController(1.0, 0.0, 0.01)
        self.__wheel_turn = PidController(0.2, 0.01, 0.01)
        self.__previous_full_speed = Point(0, 0)
        self.__previous_brake = False
        self.__is_debug = is_debug
        if is_debug:
            from debug import Plot
            self.engine_power_history = []
            self.target_engine_power_history = []
            self.speed_norm_history = []
            self.target_speed_norm_history = []
            self.wheel_turn_history = []
            self.target_wheel_turn_history = []
            self.engine_power_plot = Plot('engine power')
            self.speed_plot = Plot('speed')
            self.wheel_turn_plot = Plot('wheel turn')

    def __call__(self, direction, angle_error, engine_power, wheel_turn,
                 speed, angular_speed, speed_at_target, tick):
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
        target_engine_power = sigmoid(acceleration_error)
        engine_power_error = target_engine_power - engine_power
        engine_power_derivative = self.__engine_power(engine_power_error)
        brake = (engine_power_derivative < -self.max_engine_power_derivative and
                 not self.__previous_brake)
        self.__previous_brake = brake
        self.speed_norm_history.append(full_speed.norm())
        self.target_speed_norm_history.append(target_full_speed.norm())
        self.wheel_turn_history.append(wheel_turn)
        self.target_wheel_turn_history.append(target_wheel_turn)
        self.engine_power_history.append(engine_power)
        self.target_engine_power_history.append(target_engine_power)
        if self.__is_debug and tick % 50 == 0:
            from numpy import array
            self.speed_plot.clear()
            self.speed_plot.lines(range(len(self.speed_norm_history[:-50])),
                                  self.speed_norm_history[:-50])
            self.speed_plot.lines(range(len(self.target_speed_norm_history[:-50])),
                                  self.target_speed_norm_history[:-50])
            # self.speed_plot.lines(range(len(self.speed_norm_history)),
            #                       array(self.speed_norm_history) /
            #                       array(self.target_speed_norm_history))
            # self.speed_plot.lines(range(len(self.target_speed_norm_history)),
            #                       array(self.target_speed_norm_history) /
            #                       array(self.target_speed_norm_history))
            self.speed_plot.draw()
            self.wheel_turn_plot.clear()
            self.wheel_turn_plot.lines(range(len(self.wheel_turn_history[:-50])),
                                       self.wheel_turn_history[:-50])
            self.wheel_turn_plot.lines(range(len(self.target_wheel_turn_history[:-50])),
                                       self.target_wheel_turn_history[:-50])
            # self.wheel_turn_plot.lines(range(len(self.wheel_turn_history[:-50])),
            #                            array(self.wheel_turn_history[:-50]) /
            #                            array(self.target_wheel_turn_history[:-50]))
            # self.wheel_turn_plot.lines(range(len(self.target_wheel_turn_history[:-50])),
            #                            array(self.target_wheel_turn_history[:-50]) /
            #                            array(self.target_wheel_turn_history[:-50]))
            self.wheel_turn_plot.draw()
            self.engine_power_plot.clear()
            self.engine_power_plot.lines(range(len(self.engine_power_history[:-50])),
                                         self.engine_power_history[:-50])
            self.engine_power_plot.lines(range(len(self.target_engine_power_history[:-50])),
                                         self.target_engine_power_history[:-50])
            # self.engine_power_plot.lines(range(len(self.engine_power_history[:-50])),
            #                              array(self.engine_power_history[:-50]) /
            #                              array(self.target_engine_power_history[:-50]))
            # self.engine_power_plot.lines(range(len(self.target_engine_power_history[:-50])),
            #                              array(self.target_engine_power_history[:-50]) /
            #                              array(self.target_engine_power_history[:-50]))
            self.engine_power_plot.draw()
        return Control(engine_power_derivative, wheel_turn_derivative, brake)


class PidController:
    def __init__(self, proportional_gain, integral_gain, derivative_gain):
        self.proportional_gain = proportional_gain
        self.integral_gain = integral_gain
        self.derivative_gain = derivative_gain
        self.__previous_output = 0
        self.__previous_error = 0
        self.__integral = 0

    def __call__(self, error):
        self.__integral += error
        derivative = error - self.__previous_error
        output = (self.proportional_gain * error +
                  self.integral_gain * self.__integral +
                  self.derivative_gain * derivative)
        self.__previous_output = output
        self.__previous_error = error
        return output


def get_speed(position: Point, direction: Point, path):
    path = [position] + path

    def generate_cos():
        for i, current in islice(enumerate(path), 1, min(3, len(path) - 1)):
            yield (current - path[i - 1]).cos(path[i + 1] - current) ** 2

    course = path[1] - path[0]
    return (course / 200 +
            (course.normalized() *
             speed_gain(reduce(mul, generate_cos(), 1)) *
             course.cos(direction) ** 2))


def speed_gain(x):
    return - 3.5 / (x - 1)


def sigmoid(x):
    return 2 / (1 + exp(-x)) - 1

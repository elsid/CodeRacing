from collections import namedtuple, deque
from functools import reduce
from itertools import islice
from math import pi, exp
from operator import mul
from strategy_common import Point, normalize_angle

Control = namedtuple('Control', ('engine_power_derivative',
                                 'wheel_turn_derivative'))

History = namedtuple('History', ('current', 'target'))


class Controller:
    def __init__(self, distance_to_wheels, max_engine_power_derivative,
                 angular_speed_factor, is_debug):
        self.distance_to_wheels = distance_to_wheels
        self.max_engine_power_derivative = max_engine_power_derivative
        self.angular_speed_factor = angular_speed_factor
        self.__speed = PidController(1 / 2, 0, 1 / 8)
        self.__acceleration = PidController(1 / 4, 0, 1 / 8)
        self.__engine_power = PidController(1 / 8, 0, 1 / 8)
        self.__angle = PidController(1.0, 0, 0.0)
        self.__angular_speed_angle = PidController(1.0, 0, 0)
        self.__wheel_turn = PidController(1.0, 0, 0.0)
        self.__previous_speed = Point(0, 0)
        self.__previous_angluar_speed_angle = 0
        self.__previous_brake = False
        self.__is_debug = is_debug
        if is_debug:
            from debug import Plot

            self.history = {}
            self.plots = {}

            def plot(name, max_size=250):
                self.history[name] = History(deque(maxlen=max_size),
                                             deque(maxlen=max_size))
                self.plots[name] = Plot(name)

            def point_plots(name, max_size=250):
                # plot(name + ' x', max_size)
                # plot(name + ' y', max_size)
                plot(name + ' norm', max_size)

            point_plots('speed')
            point_plots('acceleration')
            plot('engine_power')
            plot('angle')
            plot('wheel_turn')

    def __call__(self, course, angle, direct_speed: Point, angular_speed_angle,
                 engine_power, wheel_turn, target_speed: Point, tick):
        direction = Point(1, 0).rotate(angle)
        radius = -(direction * self.distance_to_wheels).rotate(pi / 2)
        angular_speed = radius.left_orthogonal() * angular_speed_angle
        speed = direct_speed + angular_speed
        target_acceleration = self.__speed(target_speed - speed)
        tangential_acceleration = speed - self.__previous_speed
        centripetal_acceleration = (-radius *
                                    self.__previous_angluar_speed_angle ** 2)
        acceleration = tangential_acceleration + centripetal_acceleration
        acceleration_derivative = self.__acceleration(target_acceleration -
                                                      acceleration)
        target_engine_power = acceleration_derivative.norm()
        engine_power_derivative = self.__engine_power(target_engine_power -
                                                      engine_power)
        target_angle = (target_speed.absolute_rotation() +
                        course.absolute_rotation()) / 2
        angle_error = normalize_angle(target_angle - angle)
        angle_derivative = self.__angle(angle_error)
        angular_speed_angle_derivative = self.__angular_speed_angle(
            angle_derivative - angular_speed_angle)
        wheel_turn_derivative = self.__wheel_turn(
            angular_speed_angle_derivative - wheel_turn)
        self.__previous_speed = speed
        self.__previous_angluar_speed_angle = angular_speed_angle
        if self.__is_debug:

            def append_point(name, current, target):
                # append_value(name + ' x', current.x, target.x)
                # append_value(name + ' y', current.y, target.y)
                append_value(name + ' norm', current.norm(), target.norm())

            def append_value(name, current, target):
                history = self.history[name]
                history.current.append(current)
                history.target.append(target)

            append_point('speed', speed, target_speed)
            append_point('acceleration', acceleration, target_acceleration)
            append_value('engine_power', engine_power, target_engine_power)
            append_value('angle', angle, target_angle)
            append_value('wheel_turn', wheel_turn,
                         angular_speed_angle_derivative)

            if tick % 50 == 0:

                def draw(name):
                    plot = self.plots[name]
                    history = self.history[name]
                    plot.clear()
                    plot.lines(range(tick + 1 - len(history.current), tick + 1),
                               history.current)
                    plot.lines(range(tick + 1 - len(history.target), tick + 1),
                               history.target, linestyle='--')
                    plot.draw()

                def draw_point(name):
                    # draw(name + ' x')
                    # draw(name + ' y')
                    draw(name + ' norm')

                draw_point('speed')
                draw_point('acceleration')
                draw('engine_power')
                draw('angle')
                draw('wheel_turn')
        return Control(engine_power_derivative, wheel_turn_derivative)


class PidController:
    __previous_error = 0
    __integral = None

    def __init__(self, proportional_gain, integral_gain, derivative_gain):
        self.proportional_gain = proportional_gain
        self.integral_gain = integral_gain
        self.derivative_gain = derivative_gain

    def __call__(self, error):
        if self.__integral is None:
            self.__integral = error
        else:
            self.__integral += error
        derivative = error - self.__previous_error
        output = (error * self.proportional_gain +
                  self.__integral * self.integral_gain +
                  derivative * self.derivative_gain)
        self.__previous_error = error
        return output


DIRECT_FACTOR = 0.01
ANGLE_FACTOR = 4


def get_target_speed(position: Point, direction: Point, path):
    path = [position] + path

    def generate_cos():
        for i, current in islice(enumerate(path), 1, min(3, len(path) - 1)):
            yield (current - path[i - 1]).cos(path[i + 1] - current)

    course = ((path[1] - path[0]) + (path[0] - position)) / 2
    cos_product = max(1e-8 - 1, min(1 - 1e-8, reduce(mul, generate_cos(), 1)))
    return (course * DIRECT_FACTOR +
            (course.normalized() *
             speed_gain(cos_product) *
             course.cos(direction)) * ANGLE_FACTOR)


def speed_gain(x):
    return - 1 / (x - 1)


def sigmoid(x):
    return 2 / (1 + exp(-x)) - 1

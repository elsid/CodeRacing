from collections import namedtuple, deque
from functools import reduce
from itertools import islice
from math import pi, exp
from operator import mul
from strategy_common import Point, normalize_angle, Polyline

Control = namedtuple('Control', ('engine_power_derivative',
                                 'wheel_turn_derivative'))

History = namedtuple('History', ('current', 'target'))


class Controller:
    __speed = None
    __acceleration = None
    __engine_power = None
    __angle = None
    __angular_speed_angle = None
    __wheel_turn = None
    __previous_speed = None
    __previous_angular_speed_angle = None

    def __init__(self, distance_to_wheels, max_engine_power_derivative,
                 angular_speed_factor, is_debug=False):
        self.distance_to_wheels = distance_to_wheels
        self.max_engine_power_derivative = max_engine_power_derivative
        self.angular_speed_factor = angular_speed_factor
        self.reset()
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

    def reset(self):
        self.__speed = PidController(0.5, 0, 0.01)
        self.__acceleration = PidController(0.4, 0, 0.01)
        self.__engine_power = PidController(0.3, 0, 0.01)
        self.__angle = PidController(1.0, 0, 0.1)
        self.__angular_speed_angle = PidController(2.0, 0, 0.0)
        self.__wheel_turn = PidController(2.0, 0, 0.1)
        self.__previous_speed = Point(0, 0)
        self.__previous_angular_speed_angle = 0

    def __call__(self, course, angle, direct_speed: Point, angular_speed_angle,
                 engine_power, wheel_turn, target_speed: Point, tick):
        direction = Point(1, 0).rotate(angle)
        radius = -(direction * self.distance_to_wheels).rotate(pi / 2)
        angular_speed = radius.left_orthogonal() * angular_speed_angle
        speed = direct_speed + angular_speed
        target_acceleration = self.__speed(target_speed - speed)
        tangential_acceleration = speed - self.__previous_speed
        centripetal_acceleration = (-radius *
                                    self.__previous_angular_speed_angle ** 2)
        acceleration = tangential_acceleration + centripetal_acceleration
        acceleration_derivative = self.__acceleration(target_acceleration -
                                                      acceleration)
        target_engine_power = (acceleration_derivative.norm() *
                               acceleration_derivative.cos(direction))
        engine_power_derivative = self.__engine_power(
            target_engine_power - engine_power)
        if (angle < course.absolute_rotation() <
                target_speed.absolute_rotation() or
                angle > course.absolute_rotation() >
                target_speed.absolute_rotation()):
            target_angle = (target_speed.absolute_rotation() +
                            course.absolute_rotation()) / 2
        else:
            target_angle = course.absolute_rotation()
        angle_error = normalize_angle(target_angle - angle)
        angle_derivative = self.__angle(angle_error)
        angular_speed_angle_derivative = self.__angular_speed_angle(
            angle_derivative - angular_speed_angle)
        wheel_turn_derivative = self.__wheel_turn(
            angular_speed_angle_derivative - wheel_turn)
        self.__previous_speed = speed
        self.__previous_angular_speed_angle = angular_speed_angle
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
            self.__integral = self.__integral + error
        derivative = error - self.__previous_error
        output = (error * self.proportional_gain +
                  self.__integral * self.integral_gain +
                  derivative * self.derivative_gain)
        self.__previous_error = error
        return output


DIRECT_FACTOR = 1
ANGLE_FACTOR = 1.8
MAX_SPEED = 50


def generate_cos(path):
    power = len(path)
    for i, current in islice(enumerate(path), 1, len(path) - 1):
        a = current - path[i - 1]
        b = path[i + 1] - current
        yield 1 if a.norm() == 0 or b.norm() == 0 else a.cos(b) ** (power - i)


def cos_product(path):
    return reduce(mul, generate_cos(path), 1)


def get_target_speed(course: Point, path):
    factor_sum = DIRECT_FACTOR + ANGLE_FACTOR
    direct = DIRECT_FACTOR / factor_sum
    if len(path) > 2:
        angle = (ANGLE_FACTOR / factor_sum *
                 max(1e-8 - 1, min(1 - 1e-8, cos_product(path))))
    else:
        angle = 1
    if course.norm() > 0:
        return course * MAX_SPEED / course.norm() * (direct + angle)
    else:
        return Point(0, 0)


def speed_gain(x):
    return - 1 / (x - 1)


def sigmoid(x, kx=1, ky=1):
    return ky * (2 / (1 + exp(-max(-100, min(100, x / kx)))) - 1)


class StuckDetector:
    def __init__(self, history_size, stuck_distance, unstack_distance):
        self.__positions = deque(maxlen=history_size)
        self.__stuck_distance = stuck_distance
        self.__unstack_distance = unstack_distance

    def update(self, position):
        self.__positions.append(position)

    def positive_check(self):
        return (self.__positions.maxlen == len(self.__positions) and
                Polyline(self.__positions).length() < self.__stuck_distance)

    def negative_check(self):
        return Polyline(self.__positions).length() > self.__unstack_distance

    def reset(self):
        self.__positions.clear()


class DirectionDetector:
    def __init__(self, begin: Point, end: Point, min_distance):
        self.__begin = begin
        self.__end = end
        self.__min_distance = min_distance

    def update(self, position):
        if self.__end.distance(position) >= self.__min_distance:
            self.__begin = self.__end
            self.__end = position

    def __call__(self):
        return self.__end - self.__begin

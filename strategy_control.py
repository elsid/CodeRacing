from collections import namedtuple, deque
from functools import reduce
from itertools import islice
from math import pi, exp, sqrt
from operator import mul
from strategy_common import Point, normalize_angle, LimitedSum

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

    def __init__(self, distance_to_wheels, is_debug=False):
        self.distance_to_wheels = distance_to_wheels
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
        self.__speed = PidController(2, 0, 0)
        self.__acceleration = PidController(2, 0, 0)
        self.__engine_power = PidController(2, 0, 0)
        self.__angle = PidController(2, 0, 0)
        self.__angular_speed_angle = PidController(2, 0, 0)
        self.__wheel_turn = PidController(2, 0, 0)
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
        cos_val = acceleration_derivative.cos(direction)
        if -sqrt(2) / 2 < cos_val < sqrt(2) / 2:
            cos_val = sqrt(2) / 2 if cos_val >= 0 else -sqrt(2) / 2
        target_engine_power = acceleration_derivative.norm() * cos_val
        target_engine_power = max(-1, min(1, target_engine_power))
        engine_power_derivative = self.__engine_power(
            target_engine_power - engine_power)
        target_angle = course.absolute_rotation()
        direction_angle_error = normalize_angle(target_angle - angle)
        direction_angle_error = relative_angle_error(direction_angle_error)
        speed_angle_error = normalize_angle(speed.absolute_rotation() - angle)
        speed_angle_error = relative_angle_error(speed_angle_error)
        if (speed.norm() > 0 and target_speed.norm() > 0 and
                speed.cos(target_speed) < 0):
            direction_angle_error = -direction_angle_error
            speed_angle_error = -speed_angle_error
        angle_error = max(direction_angle_error, speed_angle_error, key=abs)
        angle_derivative = self.__angle(angle_error)
        target_wheel_turn = self.__angular_speed_angle(angle_derivative -
                                                       angular_speed_angle)
        target_wheel_turn = max(-1, min(1, target_wheel_turn))
        wheel_turn_derivative = self.__wheel_turn(
            target_wheel_turn - wheel_turn)
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
                         target_wheel_turn)

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


def relative_angle_error(value):
    if value > pi / 2:
        value -= pi
    elif value < -pi / 2:
        value += pi
    if pi / 4 < value < 3 * pi / 4:
        value = pi / 4 if value <= pi / 2 else 3 * pi / 4
    elif -3 * pi / 4 < value < -pi / 4:
        value = -3 * pi / 4 if value > -pi / 2 else -pi / 4
    return value


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


def generate_cos(path):
    power = len(path)
    for i, current in islice(enumerate(path), 1, len(path) - 1):
        a = current - path[i - 1]
        b = path[i + 1] - current
        yield (1 if abs(a.norm()) < 1e-3 or abs(b.norm()) < 1e-3
               else abs(a.cos(b)) ** (power - i))


def cos_product(path):
    return reduce(mul, generate_cos(path), 1)


def get_target_speed(course: Point, path, angle_to_direct_proportion,
                     max_speed):
    direct_factor = 1 / (angle_to_direct_proportion + 1)
    angle_factor = direct_factor * angle_to_direct_proportion
    if len(path) > 2:
        angle_factor *= max(1e-8 - 1, min(1 - 1e-8, cos_product(path)))
    if course.norm() > 0:
        return course * max_speed / course.norm() * (direct_factor +
                                                     angle_factor)
    else:
        return Point(0, 0)


def speed_gain(x):
    return - 1 / (x - 1)


def sigmoid(x, kx=1, ky=1):
    return ky * (2 / (1 + exp(-max(-100, min(100, x / kx)))) - 1)


class StuckDetector:
    def __init__(self, history_size, stuck_distance, unstack_distance):
        self.__distance = LimitedSum(history_size)
        self.__stuck_distance = stuck_distance
        self.__unstack_distance = unstack_distance
        self.__previous_position = None

    def update(self, position):
        if self.__previous_position is not None:
            self.__distance.update(position.distance(self.__previous_position))
        self.__previous_position = position

    def positive_check(self):
        return (self.__distance.count == self.__distance.max_count and
                self.__distance.get() < self.__stuck_distance)

    def negative_check(self):
        return self.__distance.get() > self.__unstack_distance

    def reset(self):
        self.__distance.reset()
        self.__previous_position = None


class DirectionDetector:
    def __init__(self, begin: Point, end: Point, min_distance):
        self.__begin = begin
        self.__end = end
        self.__min_distance = min_distance

    def update(self, position):
        if self.__end.distance(position) >= self.__min_distance:
            self.__begin = self.__end
            self.__end = position

    def reset(self, begin: Point, end: Point):
        self.__begin = begin
        self.__end = end

    def __call__(self):
        return self.__end - self.__begin


class CrushDetector:
    Conf = namedtuple('Conf', ('speed', 'durability'))

    def __init__(self, min_derivative):
        self.__history = deque(maxlen=2)
        self.__min_derivative = min_derivative

    def update(self, speed: Point, durability):
        self.__history.append(self.Conf(speed.norm(), durability))

    def reset(self):
        self.__history.clear()

    def check(self):
        return (self.durability_derivative() < 0 and
                self.speed_derivative() < self.__min_derivative)

    def speed_derivative(self):
        if len(self.__history) < 2:
            return 0
        return self.__history[1].speed - self.__history[0].speed

    def durability_derivative(self):
        if len(self.__history) < 2:
            return 0
        return self.__history[1].durability - self.__history[0].durability

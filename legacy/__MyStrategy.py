from collections import namedtuple
from itertools import chain
from numpy import array, arctan2, sign
from scipy.sparse.csgraph import dijkstra
from math import sqrt, cos, sin, pi, acos, exp
from itertools import islice, takewhile
from enum import Enum

from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
from model.TileType import TileType


class MyStrategy:
    controller = None
    start = None

    def move(self, me: Car, world: World, game: Game, move: Move):
        if self.controller is None:
            self.controller = Controller(
                distance_to_wheels=me.width / 4,
                max_engine_power_derivative=game.car_engine_power_change_per_tick,
                angular_speed_factor=game.car_angular_speed_factor)
        tile = current_tile(Point(me.x, me.y), game.track_tile_size)
        if world.tick < game.initial_freeze_duration_ticks:
            self.start = (tile.x, tile.y)
            return
        move.spill_oil = True
        move.throw_projectile = True
        position = Point(me.x, me.y)
        my_speed = Point(me.speed_x, me.speed_y)
        direction = Point(1, 0).rotate(me.angle)
        matrix = AdjacencyMatrix(world.tiles_x_y)
        tile_index = matrix.index(tile.x, tile.y)
        path = list(make_path(tile_index, me.next_waypoint_index, matrix,
                              world.waypoints + [self.start]))
        path = [tile_center(x, game.track_tile_size) for x in path]
        path = list(adjust_path(path, game.track_tile_size))
        path = list(shift_on_direct(path))
        path = path[1:]
        target_speed = get_speed(position, path[0], path[1], direction)
        control = self.controller(
            direction=direction,
            angle_error=me.get_angle_to(path[0].x, path[0].y),
            wheel_turn=me.wheel_turn,
            engine_power=me.engine_power,
            speed=my_speed,
            angular_speed=me.angular_speed,
            speed_at_target=target_speed,
        )
        move.engine_power += control.engine_power_derivative
        move.wheel_turn += control.wheel_turn_derivative


def shift_on_direct(path):
    if len(path) < 2:
        return (x for x in path)
    if path[0].x == path[1].x:
        last = next((i for i, p in islice(enumerate(path), 1, len(path))
                    if p.x != path[i - 1].x), len(path) - 1)
        x = path[last].x
        if x != path[0].x:
            return chain((Point(x, p.y) for p in islice(path, last)),
                         islice(path, last, len(path) - 1))
    elif path[0].y == path[1].y:
        last = next((i for i, p in islice(enumerate(path), 1, len(path))
                    if p.y != path[i - 1].y), len(path) - 1)
        y = path[last].y
        if y != path[0].y:
            return chain((Point(p.x, y) for p in islice(path, last)),
                         islice(path, last, len(path) - 1))
    return (x for x in path)


def current_tile(point, tile_size):
    return Point(tile_coord(point.x, tile_size), tile_coord(point.y, tile_size))


def tile_coord(value, tile_size):
    return int(value / tile_size)


def make_path(start_index, next_waypoint_index, matrix, waypoints):
    graph = array(matrix.values)
    _, predecessors = dijkstra(graph, return_predecessors=True)

    def generate():
        yield path(start_index, matrix.index(*waypoints[next_waypoint_index]))
        for i in range(next_waypoint_index, len(waypoints) - 1):
            src = matrix.index(*waypoints[i])
            dst = matrix.index(*waypoints[i + 1])
            yield path(src, dst)

    def path(src, dst):
        return reversed(list(back_path(src, dst)))

    def back_path(src, dst):
        while src != dst and dst >= 0:
            yield dst
            dst = predecessors.item(src, dst)

    yield Point(matrix.x_position(start_index), matrix.y_position(start_index))
    for v in chain.from_iterable(generate()):
        yield Point(matrix.x_position(v), matrix.y_position(v))


def shift_to_borders(path):
    if not path:
        return []
    for i, current in islice(enumerate(path), len(path) - 1):
        following = path[i + 1]
        direction = following - current
        yield current + direction * 0.5
    yield path[-1]


def reduce_direct_first_after_me(path):
    if len(path) < 2:
        return (x for x in path)
    following = path[0]
    after_following = path[1]
    if following.x == after_following.x or following.y == after_following.y:
        return islice(path, 1, len(path))
    return (x for x in path)


def reduce_diagonal_direct(path):
    return reduce_base_on_three(path, is_diagonal_direct)


def polar(origin, path):
    return (x.polar(origin) for x in path)


def cartesian(origin, path):
    return (x.cartesian(origin) for x in path)


def take_for_spline(path):
    if not path:
        return []

    def predicate(index, current):
        return path[index - 1].x < current.x

    yield path[0]
    generator = takewhile(lambda x: predicate(*x),
                          islice(enumerate(path), 1, len(path)))
    for _, p in generator:
        yield p


def tile_center(point, size):
    return point.map(lambda x: tile_center_coord(x, size))


def tile_center_coord(value, size):
    return (value + 0.5) * size


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


def limit(value):
    return max(-1.0, min(1.0, value))


def sigmoid(x):
    return 1 / (1 + exp(-x))

from collections import deque
from math import pi
from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
from strategy_common import Point, Polyline, get_current_tile, get_tile_center
from strategy_control import (
    Controller,
    get_target_speed,
    cos_product,
    StuckDetector,
    Control,
)
from strategy_path import (
    make_tiles_path,
    adjust_path,
    shift_on_direct,
)


class Context:
    def __init__(self, me: Car, world: World, game: Game, move: Move):
        self.me = me
        self.world = world
        self.game = game
        self.move = move

    @property
    def position(self):
        return Point(self.me.x, self.me.y)

    @property
    def speed(self):
        return Point(self.me.speed_x, self.me.speed_y)

    @property
    def direction(self):
        return Point(1, 0).rotate(self.me.angle)

    @property
    def tile(self):
        return get_current_tile(self.position, self.game.track_tile_size)


def make_release_controller(context: Context):
    return Controller(
        distance_to_wheels=context.me.width / 4,
        max_engine_power_derivative=context.game.car_engine_power_change_per_tick,
        angular_speed_factor=context.game.car_angular_speed_factor,
    )


class ReleaseStrategy:
    def __init__(self, make_controller=make_release_controller):
        self.__first_move = True
        self.__stuck = StuckDetector(history_size=100, min_distance=10)
        self.__make_controller = make_controller

    def _lazy_init(self, context: Context):
        self.__controller = self.__make_controller(context)
        self.__move_mode = MoveMode(
            start_tile=context.tile,
            controller=self.__controller,
        )

    @property
    def path(self):
        return self.__move_mode.path

    @property
    def target_position(self):
        return self.__move_mode.target_position

    def move(self, context: Context):
        if self.__first_move:
            self._lazy_init(context)
            self.__first_move = False
        self.__stuck.update(context.position)
        if self.__stuck.positive_check():
            self.__move_mode.use_backward()
        elif self.__stuck.negative_check():
            self.__move_mode.use_forward()
        self.__move_mode.move(context)


class MoveMode:
    PATH_SIZE_FOR_TARGET_SPEED = 6
    PATH_SIZE_FOR_USE_NITRO = 8

    def __init__(self, controller, start_tile):
        self.__controller = controller
        self.__path = []
        self.__tile = None
        self.__target_position = None
        self.__forward = ForwardMove(start_tile)
        self.__backward = BackwardMove(start_tile)
        self.__current = self.__forward

    @property
    def path(self):
        return self.__path

    @property
    def target_position(self):
        return self.__target_position

    def move(self, context: Context):
        self.__forward.update(context)
        self.__backward.update(context)
        if not self.__path or self.__tile != context.tile:
            self.__path = self.__current.make_path(context)
        target_position = (Polyline([context.position] + self.__path)
                           .at(context.game.track_tile_size))
        sub_path = self.__path[:self.PATH_SIZE_FOR_TARGET_SPEED]
        target_speed = get_target_speed(context.position, target_position,
                                        context.direction, sub_path)
        course = target_position - context.position
        control = self.__controller(
            course=course,
            angle=self.__current.adjust_angle(context.me.angle),
            direct_speed=context.speed,
            angular_speed_angle=context.me.angular_speed,
            engine_power=context.me.engine_power,
            wheel_turn=context.me.wheel_turn,
            target_speed=target_speed,
            tick=context.world.tick,
        )
        context.move.engine_power = (context.me.engine_power +
                                     control.engine_power_derivative)
        context.move.wheel_turn = (context.me.wheel_turn +
                                   control.wheel_turn_derivative)
        context.move.brake = self.__current.is_brake(context, control)
        context.move.spill_oil = True
        context.move.throw_projectile = True
        sub_path = self.__path[:self.PATH_SIZE_FOR_USE_NITRO]
        context.move.use_nitro = (
            len(self.__path) > 7 and
            0.95 < (cos_product([context.position] + sub_path) *
                    course.cos(context.direction)))
        self.__tile = context.tile
        self.__target_position = target_position

    def use_forward(self):
        self.__current = self.__forward

    def use_backward(self):
        self.__current = self.__forward


class ForwardMove:
    WAYPOINTS_COUNT = 5

    def __init__(self, start_tile):
        self.__start_tile = start_tile

    def make_path(self, context: Context):
        direction = Point(1, 0).rotate(context.me.angle)
        direct_speed = Point(context.me.speed_x, context.me.speed_y)
        start = context.me.next_waypoint_index
        waypoints = context.world.waypoints[start:start + self.WAYPOINTS_COUNT]
        if len(waypoints) < self.WAYPOINTS_COUNT:
            left = self.WAYPOINTS_COUNT - len(waypoints)
            waypoints += context.world.waypoints[:left]
        path = make_tiles_path(
            start_tile=self.__start_tile,
            waypoints=waypoints,
            tiles=context.world.tiles_x_y,
            direction=direct_speed + direction,
        )
        path = [get_tile_center(x, context.game.track_tile_size) for x in path]
        shift = (context.game.track_tile_size / 2 -
                 context.game.track_tile_margin -
                 1.5 * max(context.me.width, context.me.height) / 2)
        path = list(adjust_path(path, shift))
        path = list(shift_on_direct(path))
        return [(path[1] + path[2]) / 2] + path[2:]

    def update(self, context: Context):
        self.__start_tile = context.tile

    @staticmethod
    def is_brake(context: Context, control: Control):
        return (-context.game.car_engine_power_change_per_tick >
                control.engine_power_derivative)

    @staticmethod
    def adjust_angle(value):
        return value


class BackwardMove:
    def __init__(self, start_tile):
        self.__tile_history = deque([start_tile], maxlen=10)

    def make_path(self, context: Context):
        return ([get_tile_center(x, context.game.track_tile_size)
                 for x in reversed(self.__tile_history)])

    def update(self, context: Context):
        position = Point(context.me.x, context.me.y)
        tile = get_current_tile(position, context.game.track_tile_size)
        if tile != self.__tile_history[-1]:
            self.__tile_history.append(tile)

    @staticmethod
    def is_brake(context: Context, control: Control):
        return (context.game.car_engine_power_change_per_tick <
                control.engine_power_derivative)

    @staticmethod
    def adjust_angle(value):
        return value - pi if value > 0 else value + pi

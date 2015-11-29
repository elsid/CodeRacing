from enum import Enum
from collections import deque
from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
from strategy_common import Point, Polyline, get_current_tile, get_tile_center
from strategy_control import Controller, get_target_speed, cos_product
from strategy_path import (
    make_tiles_path,
    adjust_path,
    shift_on_direct,
)


class MoveMode(Enum):
    FORWARD = 1
    BACKWARD = -1


class ReleaseStrategy:
    controller = None
    start = None
    path = None
    tiles_path = None
    previous_tile = None
    position = None
    target_position = None
    middle = None

    def __init__(self):
        self.position_history = deque(maxlen=100)
        self.tile_history = deque(maxlen=1)
        self.move_mode = MoveMode.FORWARD

    def move(self, me: Car, world: World, game: Game, move: Move,
             is_debug=False):
        if self.controller is None:
            self.controller = Controller(
                distance_to_wheels=me.width / 4,
                max_engine_power_derivative=game.car_engine_power_change_per_tick,
                angular_speed_factor=game.car_angular_speed_factor,
                is_debug=is_debug,
            )
        position = Point(me.x, me.y)
        speed = Point(me.speed_x, me.speed_y)
        tile = get_current_tile(position, game.track_tile_size)
        if world.tick < game.initial_freeze_duration_ticks:
            self.start = (tile.x, tile.y)
            self.previous_tile = tile
            self.position = position
            return
        if self.path is None or tile != self.previous_tile:
            if self.move_mode != MoveMode.FORWARD:
                self.move_forward(me, world, game)
            else:
                self.build_forward_path(me, world, game)
        if self.position_history.maxlen == len(self.position_history):
            if Polyline(self.position_history).length() < 5:
                if self.move_mode == MoveMode.FORWARD:
                    self.move_backward(game)
                elif self.move_mode == MoveMode.BACKWARD:
                    self.move_forward(me, world, game)
                self.position_history.clear()
            elif self.move_mode == MoveMode.BACKWARD:
                self.move_forward(me, world, game)
        direction = Point(1, 0).rotate(me.angle)
        direct_speed = Point(me.speed_x, me.speed_y)
        target_position = (Polyline([position] + self.path)
                           .at(game.track_tile_size))
        target_speed = get_target_speed(position, target_position,
                                        direction, self.path)
        course = target_position - position
        control = self.controller(
            course=course,
            angle=me.angle,
            direct_speed=direct_speed,
            angular_speed_angle=me.angular_speed,
            engine_power=me.engine_power,
            wheel_turn=me.wheel_turn,
            target_speed=target_speed,
            tick=world.tick,
        )
        move.engine_power = me.engine_power + control.engine_power_derivative
        move.wheel_turn = me.wheel_turn + control.wheel_turn_derivative
        if speed.norm() / target_speed.norm() > 1:
            if self.move_mode == MoveMode.FORWARD:
                move.brake = (-game.car_engine_power_change_per_tick >
                              control.engine_power_derivative)
            elif self.move_mode == MoveMode.BACKWARD:
                move.brake = (game.car_engine_power_change_per_tick <
                              control.engine_power_derivative)
        move.spill_oil = True
        move.throw_projectile = True
        move.use_nitro = (len(self.path) > 7 and 0.95 <
                          cos_product([position] + self.path[:8]) *
                          course.cos(direction))
        if self.previous_tile != tile:
            self.tile_history.append(tile)
        self.previous_tile = tile
        self.position = position
        self.target_position = target_position
        self.position_history.append(position)

    def move_forward(self, me: Car, world: World, game: Game):
        self.move_mode = MoveMode.FORWARD
        self.controller.reset()
        self.build_forward_path(me, world, game)

    def move_backward(self, game: Game):
        self.move_mode = MoveMode.BACKWARD
        self.controller.reset()
        self.build_backward_path(game)

    def build_forward_path(self, me: Car, world: World, game: Game):
        self.move_mode = MoveMode.FORWARD
        direction = Point(1, 0).rotate(me.angle)
        direct_speed = Point(me.speed_x, me.speed_y)
        path = list(make_tiles_path(
            start_tile=self.previous_tile,
            waypoints=world.waypoints + [self.start],
            next_waypoint_index=me.next_waypoint_index,
            tiles=world.tiles_x_y,
            direction=direct_speed + direction,
        ))
        path = [get_tile_center(x, game.track_tile_size) for x in path]
        self.tiles_path = path
        shift = (game.track_tile_size / 2 -
                 game.track_tile_margin - 1.5 * max(me.width, me.height) / 2)
        path = list(adjust_path(path, shift))
        path = list(shift_on_direct(path))
        self.path = [(path[1] + path[2]) / 2] + path[2:]

    def build_backward_path(self, game: Game):
        self.path = ([get_tile_center(x, game.track_tile_size)
                      for x in reversed(self.tile_history)])

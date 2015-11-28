from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
from strategy_common import Point, get_current_tile, get_tile_center
from strategy_control import Controller, get_target_speed
from strategy_path import (
    make_tiles_path,
    adjust_path,
    shift_on_direct,
    reduce_direct,
)


class ReleaseStrategy:
    controller = None
    start = None
    path = None
    tiles_path = None
    previous_tile = None

    def move(self, me: Car, world: World, game: Game, move: Move,
             is_debug=False):
        if self.controller is None:
            self.controller = Controller(
                distance_to_wheels=me.width / 4,
                max_engine_power_derivative=game.car_engine_power_change_per_tick,
                angular_speed_factor=game.car_angular_speed_factor,
                is_debug=is_debug)
        tile = get_current_tile(Point(me.x, me.y), game.track_tile_size)
        if world.tick < game.initial_freeze_duration_ticks:
            self.start = (tile.x, tile.y)
            self.previous_tile = tile
            return
        move.spill_oil = True
        move.throw_projectile = True
        position = Point(me.x, me.y)
        direct_speed = Point(me.speed_x, me.speed_y)
        direction = Point(1, 0).rotate(me.angle)
        path = list(make_tiles_path(
            start=self.start,
            start_tile=self.previous_tile,
            waypoints=world.waypoints,
            next_waypoint_index=me.next_waypoint_index,
            tiles=world.tiles_x_y,
        ))
        path = [get_tile_center(x, game.track_tile_size) for x in path]
        self.tiles_path = path
        path = list(adjust_path(path, game.track_tile_size))
        path = list(reduce_direct(path))
        path = list(shift_on_direct(path))
        path = path[2:]
        target_speed = get_target_speed(position, direction, path)
        control = self.controller(
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
        move.brake = (-game.car_engine_power_change_per_tick >
                      control.engine_power_derivative)
        self.path = [position] + path
        self.previous_tile = tile

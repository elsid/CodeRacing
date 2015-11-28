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


class ReleaseStrategy:
    controller = None
    start = None
    path = None
    tiles_path = None
    previous_tile = None
    position = None
    target_position = None
    middle = None

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
        direction = Point(1, 0).rotate(me.angle)
        direct_speed = Point(me.speed_x, me.speed_y)
        if self.path is None or tile != self.previous_tile:
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
                     game.track_tile_margin - max(me.width, me.height) / 2)
            path = list(adjust_path(path, shift))
            path = list(shift_on_direct(path))
            self.path = [(path[1] + path[2]) / 2] + path[2:]
        position = Point(me.x, me.y)
        target_position = (Polyline([position] + self.path)
                           .at(game.track_tile_size))
        course = target_position - position
        target_speed = get_target_speed(position, target_position, direction,
                                        self.path)
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
        move.brake = (-game.car_engine_power_change_per_tick >
                      control.engine_power_derivative)
        move.spill_oil = True
        move.throw_projectile = True
        move.use_nitro = (0.95 <
                          cos_product([position] + self.path[:4]) *
                          course.cos(direction))
        self.previous_tile = tile
        self.position = position
        self.target_position = target_position

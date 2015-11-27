from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
from strategy.common import Point, get_current_tile, get_tile_center
from strategy.control import Controller, get_speed
from strategy.path import make_tiles_path, adjust_path, shift_on_direct


class ReleaseStrategy:
    controller = None
    start = None
    path = None

    def move(self, me: Car, world: World, game: Game, move: Move):
        if self.controller is None:
            self.controller = Controller(
                distance_to_wheels=me.width / 4,
                max_engine_power_derivative=game.car_engine_power_change_per_tick,
                angular_speed_factor=game.car_angular_speed_factor)
        tile = get_current_tile(Point(me.x, me.y), game.track_tile_size)
        if world.tick < game.initial_freeze_duration_ticks:
            self.start = (tile.x, tile.y)
            return
        move.spill_oil = True
        move.throw_projectile = True
        position = Point(me.x, me.y)
        my_speed = Point(me.speed_x, me.speed_y)
        direction = Point(1, 0).rotate(me.angle)
        path = list(make_tiles_path(
            start=self.start,
            position=position,
            waypoints=world.waypoints,
            next_waypoint_index=me.next_waypoint_index,
            tile_size=game.track_tile_size,
            tiles=world.tiles_x_y
        ))
        path = [get_tile_center(x, game.track_tile_size) for x in path]
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
        self.path = path

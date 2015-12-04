from collections import deque
from math import cos, pi
from itertools import chain
from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
from strategy_common import Point, Polyline, get_current_tile, Line
from strategy_control import (
    Controller,
    get_target_speed,
    cos_product,
    StuckDetector,
    DirectionDetector,
)
from strategy_path import (
    make_tiles_path,
    adjust_path,
    shift_on_direct,
    get_point_index,
)
from strategy_barriers import make_tiles_barriers


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
        self.__make_controller = make_controller

    def __lazy_init(self, context: Context):
        self.__stuck = StuckDetector(
            history_size=300,
            stuck_distance=min(context.me.width, context.me.height) / 5,
            unstack_distance=context.game.track_tile_size / 2,
        )
        self.__direction = DirectionDetector(
            begin=context.position,
            end=context.position + context.direction,
            min_distance=max(context.me.width, context.me.height),
        )
        self.__controller = self.__make_controller(context)
        self.__move_mode = MoveMode(
            start_tile=context.tile,
            controller=self.__controller,
            get_direction=self.__direction,
            waypoints_count=(len(context.world.waypoints) *
                             context.game.lap_count)
        )

    @property
    def path(self):
        return self.__move_mode.path

    @property
    def target_position(self):
        return self.__move_mode.target_position

    def move(self, context: Context):
        if self.__first_move:
            self.__lazy_init(context)
            self.__first_move = False
        if context.world.tick > context.game.initial_freeze_duration_ticks:
            self.__stuck.update(context.position)
            self.__direction.update(context.position)
        if self.__stuck.positive_check():
            self.__move_mode.switch()
            self.__stuck.reset()
            self.__controller.reset()
        elif self.__move_mode.is_backward() and self.__stuck.negative_check():
            self.__move_mode.use_forward()
            self.__stuck.reset()
            self.__controller.reset()
        self.__move_mode.move(context)


class MoveMode:
    PATH_SIZE_FOR_TARGET_SPEED = 3
    PATH_SIZE_FOR_USE_NITRO = 5
    FORWARD_WAYPOINTS_COUNT = 5
    BACKWARD_WAYPOINTS_COUNT = 3

    def __init__(self, controller, start_tile, get_direction, waypoints_count):
        self.__controller = controller
        self.__path = []
        self.__tile = None
        self.__target_position = None
        self.__forward = ForwardPathBuilder(
            start_tile=start_tile,
            get_direction=get_direction,
            waypoints_count=waypoints_count,
        )
        self.__backward = BackwardPathBuilder(
            start_tile=start_tile,
            get_direction=get_direction,
            waypoints_count=self.BACKWARD_WAYPOINTS_COUNT,
        )
        self.__current = self.__forward
        self.__get_direction = get_direction
        self.__course = Course()

    @property
    def path(self):
        return self.__path

    @property
    def target_position(self):
        return self.__target_position

    def is_backward(self):
        return self.__current == self.__backward

    def move(self, context: Context):
        self.__update_path(context)
        course = self.__course.get(context, self.__path)
        target_position = context.position + course
        speed_path = self.__path[:self.PATH_SIZE_FOR_TARGET_SPEED]
        target_speed = get_target_speed(context.position, target_position,
                                        speed_path)
        if target_speed.norm() == 0:
            return
        control = self.__controller(
            course=course,
            angle=context.me.angle,
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
        if (target_speed.norm() == 0 or
            (context.speed.norm() > 0 and
             (context.speed.norm() > target_speed.norm() and
              context.speed.cos(target_speed) >= 0 or
              context.speed.cos(target_speed) < 0))):
            if context.speed.cos(context.direction) >= 0:
                context.move.brake = (
                    -context.game.car_engine_power_change_per_tick >
                    control.engine_power_derivative)
            else:
                context.move.brake = (
                    context.game.car_engine_power_change_per_tick >
                    control.engine_power_derivative)
        context.move.spill_oil = True
        context.move.throw_projectile = True
        nitro_path = ([context.position] +
                      self.__path[:self.PATH_SIZE_FOR_USE_NITRO])
        context.move.use_nitro = (
            context.world.tick > context.game.initial_freeze_duration_ticks and
            len(nitro_path) >= self.PATH_SIZE_FOR_USE_NITRO + 1 and
            0.95 < (cos_product(nitro_path)))
        self.__target_position = target_position
        self.__tile = context.tile
        self.__backward.start_tile = context.tile
        self.__forward.start_tile = context.tile

    def use_forward(self):
        self.__current = self.__forward
        self.__path.clear()

    def use_backward(self):
        self.__current = self.__backward
        self.__path.clear()

    def switch(self):
        if self.__current == self.__forward:
            self.use_backward()
        else:
            self.use_forward()

    def __update_path(self, context: Context):
        if (not self.__path or
                context.position.distance(self.__path[0]) >
                2 * context.game.track_tile_size):
            self.__path = self.__current.make(context)
        while (self.__path and
               context.position.distance(self.__path[0]) <
               0.75 * context.game.track_tile_size):
            self.__path = self.__path[1:]


class BasePathBuilder:
    def __init__(self, start_tile, get_direction):
        self.start_tile = start_tile
        self.get_direction = get_direction

    def make(self, context: Context):
        waypoints = self._waypoints(context.me.next_waypoint_index,
                                    context.world.waypoints)
        direction = self.get_direction().normalized()
        if context.speed.norm() > 0:
            direction = direction + context.speed.normalized()
        path = list(make_tiles_path(
            start_tile=context.tile,
            waypoints=waypoints,
            tiles=context.world.tiles_x_y,
            direction=direction,
        ))
        if self.start_tile != path[0]:
            path = [self.start_tile] + path
        path = [(x + Point(0.5, 0.5)) * context.game.track_tile_size
                for x in path]
        shift = (context.game.track_tile_size / 2 -
                 context.game.track_tile_margin -
                 max(context.me.width, context.me.height))
        path = list(adjust_path(path, shift))
        path = list(shift_on_direct(path))
        return path

    def _waypoints(self, next_waypoint_index, waypoints):
        raise NotImplementedError()


class ForwardPathBuilder(BasePathBuilder):
    def __init__(self, start_tile, get_direction, waypoints_count):
        super().__init__(start_tile, get_direction)
        self.__waypoints_count = waypoints_count

    # def make(self, context: Context):
    #     result = super().make(context)
    #     return [(result[1] + result[2]) / 2] + result[2:]

    def _waypoints(self, next_waypoint_index, waypoints):
        end = next_waypoint_index + self.__waypoints_count
        result = waypoints[next_waypoint_index:end]
        left = self.__waypoints_count - len(result)
        while left > 0:
            add = waypoints[:left]
            result += add
            left -= len(add)
        return result


class BackwardPathBuilder(BasePathBuilder):
    def __init__(self, start_tile, get_direction, waypoints_count):
        super().__init__(start_tile, get_direction)
        self.__waypoints_count = waypoints_count
        self.__begin = None

    def make(self, context: Context):
        result = super().make(context)[1:]
        if context.speed.norm() < 1:
            first = (context.position -
                     self.get_direction().normalized() *
                     context.game.track_tile_size)
            result = [first] + result
        return result

    def _waypoints(self, next_waypoint_index, waypoints):
        if self.__begin is None:
            self.__begin = (next_waypoint_index - 1) % len(waypoints)
        self.__begin = next((i for i, v in enumerate(waypoints)
                             if Point(v[0], v[1]) == self.start_tile),
                            self.__begin)
        begin = self.__begin
        result = list(reversed(waypoints[:begin][-self.__waypoints_count:]))
        left = self.__waypoints_count - len(result)
        while left > 0:
            add = list(reversed(waypoints[-left:]))
            result += add
            left -= len(add)
        return result


class Course:
    def __init__(self):
        self.__tile_barriers = None

    def get(self, context: Context, path):
        if self.__tile_barriers is None:
            self.__tile_barriers = make_tiles_barriers(
                tiles=context.world.tiles_x_y,
                margin=context.game.track_tile_margin,
                size=context.game.track_tile_size,
            )
        tile_size = context.game.track_tile_size
        target_position = (Polyline([context.position] + path)
                           .at(0.75 * tile_size))
        course = target_position - context.position
        current_tile = context.tile
        target_tile = get_current_tile(target_position, tile_size)
        range_x = (range(current_tile.x, target_tile.x)
                   if current_tile.x <= target_tile.x
                   else range(target_tile.x, current_tile.x))
        range_y = (range(current_tile.y, target_tile.y)
                   if current_tile.y <= target_tile.y
                   else range(target_tile.y, current_tile.y))

        def generate_tiles():
            for x in range_x:
                for y in range_y:
                    yield Point(x, y)

        row_size = len(context.world.tiles_x_y[0])

        def generate_barriers():
            for tile in generate_tiles():
                yield self.__tile_barriers[get_point_index(tile, row_size)]

        barriers = list(chain.from_iterable(generate_barriers()))

        def has_intersection(current_angle):
            end = context.position + course.rotate(current_angle)
            line = Line(context.position, end)
            return next((1 for x in barriers if x.has_intersection(line)), 0)

        angle = find_false(-cos(1), cos(1), has_intersection, 0.1)
        if angle is not None:
            return course.rotate(angle)
        angle = find_false(-cos(1) - pi, cos(1) - pi, has_intersection, 0.1)
        if angle is not None:
            return course.rotate(angle)
        return course


def find_false(begin, end, function, min_interval):
    queue = deque([(begin, end)])
    while queue:
        begin, end = queue.popleft()
        if end - begin < min_interval:
            return None
        middle = (begin + end) / 2
        if not function(middle):
            return middle
        queue.append((begin, middle))
        queue.append((middle, end))
    return None

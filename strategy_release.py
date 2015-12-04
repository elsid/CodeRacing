from collections import deque, namedtuple
from math import cos, pi
from itertools import chain
from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
from model.CarType import CarType
from strategy_common import Point, Polyline, get_current_tile
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
from strategy_barriers import (
    make_tiles_barriers,
    make_units_barriers,
    make_has_intersection_with_line,
    make_has_intersection_with_lane,
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
        self.__make_controller = make_controller

    def __lazy_init(self, context: Context):
        self.__stuck = StuckDetector(
            history_size=250,
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
        if context.world.tick < context.game.initial_freeze_duration_ticks:
            context.me.engine_power = 1
            return
        self.__stuck.update(context.position)
        self.__direction.update(context.position)
        if self.__stuck.positive_check():
            self.__move_mode.switch()
            self.__stuck.reset()
            self.__controller.reset()
        elif (not self.__move_mode.is_forward() and
              self.__stuck.negative_check()):
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
        self.__forward = ForwardWaypointsPathBuilder(
            start_tile=start_tile,
            get_direction=get_direction,
            waypoints_count=waypoints_count,
        )
        self.__backward = BackwardWaypointsPathBuilder(
            start_tile=start_tile,
            get_direction=get_direction,
            waypoints_count=self.BACKWARD_WAYPOINTS_COUNT,
        )
        self.__unstuck = UnstuckPathBuilder()
        self.__states = {
            id(self.__forward): self.__backward,
            id(self.__backward): self.__unstuck,
            id(self.__unstuck): self.__forward,
        }
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

    def is_forward(self):
        return self.__current == self.__forward

    def move(self, context: Context):
        self.__update_path(context)
        course = self.__course.get(context, self.__path)
        target_position = context.position + course
        speed_path = ([context.position - self.__get_direction(),
                      context.position] +
                      self.__path[:self.PATH_SIZE_FOR_TARGET_SPEED])
        target_speed = get_target_speed(course, speed_path)
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
                    control.engine_power_derivative / 2)
            else:
                context.move.brake = (
                    context.game.car_engine_power_change_per_tick >
                    control.engine_power_derivative / 2)
        context.move.spill_oil = (
            context.me.oil_canister_count > 1 or
            make_has_intersection_with_line(
                position=context.position,
                course=(-context.direction * context.game.track_tile_size),
                barriers=list(generate_cars_barriers(context)),
            )(0))
        context.move.throw_projectile = make_has_intersection_with_lane(
            position=context.position,
            course=(context.direction * context.game.track_tile_size),
            barriers=list(generate_cars_barriers(context)),
            width=(context.game.washer_radius
                   if context.me.type == CarType.BUGGY
                   else context.game.tire_radius)
        )(0)
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

    def use_unstuck(self):
        self.__current = self.__unstuck
        self.__path.clear()

    def switch(self):
        self.__current = self.__states[id(self.__current)]
        self.__path.clear()

    def __update_path(self, context: Context):
        if (not self.__path or
                context.position.distance(self.__path[0]) >
                2 * context.game.track_tile_size):
            self.__path = self.__current.make(context)
        while (self.__path and
               context.position.distance(self.__path[0]) <
               0.75 * context.game.track_tile_size):
            self.__path = self.__path[1:]


class WaypointsPathBuilder:
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
                 max(context.me.width, context.me.height) / 2)
        path = list(adjust_path(path, shift))
        path = list(shift_on_direct(path))
        return path

    def _waypoints(self, next_waypoint_index, waypoints):
        raise NotImplementedError()


class ForwardWaypointsPathBuilder(WaypointsPathBuilder):
    def __init__(self, start_tile, get_direction, waypoints_count):
        super().__init__(start_tile, get_direction)
        self.__waypoints_count = waypoints_count

    def _waypoints(self, next_waypoint_index, waypoints):
        end = next_waypoint_index + self.__waypoints_count
        result = waypoints[next_waypoint_index:end]
        left = self.__waypoints_count - len(result)
        while left > 0:
            add = waypoints[:left]
            result += add
            left -= len(add)
        return result


class BackwardWaypointsPathBuilder(WaypointsPathBuilder):
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


class UnstuckPathBuilder:
    def make(self, context: Context):
        return [context.position + context.direction *
                0.9 * context.game.track_tile_size]


Params = namedtuple('Params', ('course', 'barriers', 'width',
                               'make_has_intersection'))


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
        target_position = Polyline([context.position] + path).at(tile_size)
        course = target_position - context.position
        current_tile = context.tile
        target_tile = get_current_tile(target_position, tile_size)
        range_x = list(range(current_tile.x, target_tile.x + 1)
                       if current_tile.x <= target_tile.x
                       else range(target_tile.x, current_tile.x + 1))
        range_y = list(range(current_tile.y, target_tile.y + 1)
                       if current_tile.y <= target_tile.y
                       else range(target_tile.y, current_tile.y + 1))

        def generate_tiles():
            for x in range_x:
                for y in range_y:
                    yield Point(x, y)

        tiles = list(generate_tiles())
        row_size = len(context.world.tiles_x_y[0])

        def generate_tiles_barriers():
            def impl():
                for tile in tiles:
                    yield self.__tile_barriers[get_point_index(tile, row_size)]
            return chain.from_iterable(impl())

        tiles_barriers = list(generate_tiles_barriers())
        all_barriers = list(chain(tiles_barriers,
                                  generate_units_barriers(context)))
        width = max(context.me.width, context.me.height)
        params = [
            Params(course=course, barriers=all_barriers, width=width,
                   make_has_intersection=make_has_intersection_with_lane),
            Params(course=course, barriers=all_barriers, width=None,
                   make_has_intersection=make_has_intersection_with_line),
            Params(course=course, barriers=tiles_barriers, width=width,
                   make_has_intersection=make_has_intersection_with_lane),
            Params(course=course, barriers=tiles_barriers, width=None,
                   make_has_intersection=make_has_intersection_with_line),
            Params(course=-course, barriers=all_barriers, width=width,
                   make_has_intersection=make_has_intersection_with_lane),
            Params(course=-course, barriers=all_barriers, width=None,
                   make_has_intersection=make_has_intersection_with_line),
            Params(course=-course, barriers=tiles_barriers, width=width,
                   make_has_intersection=make_has_intersection_with_lane),
            Params(course=-course, barriers=tiles_barriers, width=None,
                   make_has_intersection=make_has_intersection_with_line),
        ]
        for x in params:
            has_intersection = (
                x.make_has_intersection(context.position, x.course, x.barriers)
                if x.width is None else
                x.make_has_intersection(context.position, x.course, x.barriers,
                                        x.width)
            )
            new_course = adjust_course(x.course, has_intersection)
            if new_course is not None:
                return -new_course if x.course == -course else new_course
        return course


def generate_units_barriers(context: Context):
    return chain.from_iterable([
        make_units_barriers(context.world.projectiles),
        generate_cars_barriers(context),
    ])


def generate_cars_barriers(context: Context):
    cars = (x for x in context.world.cars if x.id != context.me.id)
    return make_units_barriers(cars)


def adjust_course(course, has_intersection):
    angle = find_false(-cos(1), cos(1), has_intersection, 2 ** -4)
    if angle is not None:
        return course.rotate(angle)
    return None


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

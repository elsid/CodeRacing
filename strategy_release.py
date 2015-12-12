from collections import deque, namedtuple
from copy import copy
from math import cos, radians, pi
from itertools import chain, islice
from functools import reduce
from operator import mul
from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
from model.CarType import CarType
from model.TileType import TileType
from strategy_common import (
    Point,
    Polyline,
    get_current_tile,
    LimitedSum,
    Line,
    Curve,
    get_tile_center,
)
from strategy_control import (
    Controller,
    get_target_speed,
    cos_product,
    StuckDetector,
    DirectionDetector,
    CrushDetector,
)
from strategy_path import (
    make_tiles_path,
    adjust_path,
    shift_on_direct,
    get_point_index,
    adjust_for_bonuses,
    PriorityConf,
)
from strategy_barriers import (
    make_tiles_barriers,
    make_units_barriers,
    make_has_intersection_with_line,
    make_has_intersection_with_lane,
    Rectangle,
)


BUGGY_INITIAL_ANGLE_TO_DIRECT_PROPORTION = 2.4
JEEP_INITIAL_ANGLE_TO_DIRECT_PROPORTION = 2.6
SPEED_LOSS_HISTORY_SIZE = 500
MIN_SPEED_LOSS = 1 / SPEED_LOSS_HISTORY_SIZE
MAX_SPEED_LOSS = 1 / SPEED_LOSS_HISTORY_SIZE
CHANGE_PER_TICKS_COUNT = SPEED_LOSS_HISTORY_SIZE / 5
MAX_CANISTER_COUNT = 1
COURSE_PATH_SIZE = 3
TARGET_SPEED_PATH_SIZE = 3
BUGGY_PATH_SIZE_FOR_USE_NITRO = 5
JEEP_PATH_SIZE_FOR_USE_NITRO = 5
MAX_SPEED = 50
MAX_SPEED_THROUGH_UNKNOWN = 30
PATH_SIZE_FOR_BONUSES = 5
CAR_SPEED_FACTOR = 1.2
WASHER_INTERVAL = 3
TIRE_INTERVAL = 2


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

    @property
    def tile_index(self):
        return get_point_index(self.tile, self.world.height)

    @property
    def is_buggy(self):
        return self.me.type == CarType.BUGGY

    @property
    def world_rectangle(self):
        return Rectangle(left_top=Point(0, 0),
                         right_bottom=Point(self.world.width,
                                            self.world.height))

    @property
    def opponents_cars(self):
        return (x for x in self.world.cars if not x.teammate)


def make_release_controller(context: Context):
    return Controller(distance_to_wheels=context.me.width / 4)


def tiles_has_unknown(tiles):
    for column in tiles:
        for tile in column:
            if tile == TileType.UNKNOWN:
                return True


class ReleaseStrategy:
    def __init__(self, make_controller=make_release_controller):
        self.__first_move = True
        self.__make_controller = make_controller

    def __lazy_init(self, context: Context):
        self.__stuck = StuckDetector(
            history_size=150,
            stuck_distance=min(context.me.width, context.me.height),
            unstack_distance=max(context.me.width, context.me.height) * 1.5,
        )
        self.__direction = DirectionDetector(
            begin=context.position,
            end=context.position + context.direction,
            min_distance=max(context.me.width, context.me.height),
        )
        self.__controller = self.__make_controller(context)
        self.__move_mode = AdaptiveMoveMode(
            start_tile=context.tile,
            controller=self.__controller,
            get_direction=self.__direction,
            waypoints_count=(len(context.world.waypoints) *
                             context.game.lap_count),
            speed_angle_to_direct_proportion=(
                BUGGY_INITIAL_ANGLE_TO_DIRECT_PROPORTION
                if context.is_buggy
                else JEEP_INITIAL_ANGLE_TO_DIRECT_PROPORTION
            )
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
        context.move.engine_power = 1
        if context.world.tick < context.game.initial_freeze_duration_ticks:
            return
        if context.me.durability > 0:
            self.__stuck.update(context.position)
        else:
            self.__stuck.reset()
        self.__direction.update(context.position)
        if self.__stuck.positive_check():
            self.__move_mode.on_stuck()
            self.__stuck.reset()
            self.__controller.reset()
            self.__direction.reset(begin=context.position,
                                   end=context.position + context.direction)
        elif not self.__move_mode.is_main and self.__stuck.negative_check():
            self.__move_mode.use_main()
            self.__stuck.reset()
            self.__controller.reset()
            self.__direction.reset(begin=context.position,
                                   end=context.position + context.direction)
        elif (self.__move_mode.is_main and context.speed.norm() > 0 and
                context.direction.cos(context.speed) < -cos(1)):
            self.__move_mode.on_backward()
        self.__move_mode.move(context)


class AdaptiveMoveMode:
    def __init__(self, controller, start_tile, get_direction, waypoints_count,
                 speed_angle_to_direct_proportion):
        self.__current_index = 0
        self.__move_mode = MoveMode(
            start_tile=start_tile,
            controller=controller,
            get_direction=get_direction,
            waypoints_count=waypoints_count,
            speed_angle_to_direct_proportion=speed_angle_to_direct_proportion,
        )
        self.__crush = CrushDetector(min_derivative=-0.6)
        self.__speed_loss = SpeedLoss(history_size=SPEED_LOSS_HISTORY_SIZE)
        self.__last_change = 0

    @property
    def path(self):
        return self.__move_mode.path

    @property
    def target_position(self):
        return self.__move_mode.target_position

    @property
    def is_main(self):
        return self.__move_mode.is_main

    def move(self, context: Context):
        self.__crush.update(context.speed, context.me.durability)
        self.__speed_loss.update(self.__crush.check(),
                                 -self.__crush.speed_derivative())
        speed_loss = self.__speed_loss.get()
        if speed_loss > MAX_SPEED_LOSS:
            self.__change(1 / 0.999, context.world.tick)
        elif speed_loss < MIN_SPEED_LOSS:
            self.__change(0.999, context.world.tick)
        self.__move_mode.move(context)

    def use_main(self):
        self.__move_mode.use_main()

    def on_stuck(self):
        self.__move_mode.on_stuck()

    def on_backward(self):
        self.__move_mode.on_backward()

    def __change(self, value, tick):
        if tick - self.__last_change >= CHANGE_PER_TICKS_COUNT:
            self.__move_mode.speed_angle_to_direct_proportion *= value
            self.__last_change = tick


class MoveMode:
    def __init__(self, controller, start_tile, get_direction, waypoints_count,
                 speed_angle_to_direct_proportion):
        self.__controller = controller
        self.__path = Path(
            start_tile=start_tile,
            get_direction=get_direction,
            waypoints_count=waypoints_count,
        )
        self.__target_position = None
        self.__get_direction = get_direction
        self.__course = Course()
        self.__previous_path = []
        self.speed_angle_to_direct_proportion = speed_angle_to_direct_proportion

    @property
    def path(self):
        return self.__previous_path

    @property
    def target_position(self):
        return self.__target_position

    @property
    def is_main(self):
        return self.__path.is_main

    def move(self, context: Context):
        path = self.__path.get(context)
        course = self.__course.get(context, path[:COURSE_PATH_SIZE])
        speed_path = ([context.position - self.__get_direction(),
                       context.position] + path[:TARGET_SPEED_PATH_SIZE])
        max_speed = (
            MAX_SPEED_THROUGH_UNKNOWN
            if path_has_tiles(path, context.world.tiles_x_y,
                              context.game.track_tile_size,
                              TileType.UNKNOWN)
            else MAX_SPEED
        )
        target_speed = get_target_speed(
            course=course,
            path=speed_path,
            angle_to_direct_proportion=(self.speed_angle_to_direct_proportion *
                                        max_speed / MAX_SPEED),
            max_speed=max_speed,
        )
        self.__target_position = context.position + course
        self.__previous_path = path
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
            backward=not self.__path.is_main
        )
        context.move.engine_power = control.engine_power
        context.move.wheel_turn = control.wheel_turn
        context.move.brake = control.brake
        context.move.spill_oil = (
            context.me.oil_canister_count > MAX_CANISTER_COUNT or
            make_has_intersection_with_line(
                position=context.position,
                course=(-context.direction * context.game.track_tile_size),
                barriers=list(generate_opponents_cars_barriers(context)),
            )(0))
        context.move.throw_projectile = throw_projectile(
            context, self.__course.tile_barriers)
        nitro_path_size = (
            BUGGY_PATH_SIZE_FOR_USE_NITRO if context.is_buggy
            else JEEP_PATH_SIZE_FOR_USE_NITRO
        )
        if context.world.tick == context.game.initial_freeze_duration_ticks + 1:
            nitro_path = path[:nitro_path_size]
        else:
            nitro_path = ([context.position - self.__get_direction(),
                           context.position] + path[:nitro_path_size])
        if (context.world.tick > context.game.initial_freeze_duration_ticks and
                len(nitro_path) >= nitro_path_size):
            nitro_cos = cos_product(nitro_path)
            context.move.use_nitro = (
                nitro_cos > 0.9 or nitro_cos > 0.6 and
                target_speed.norm() - context.speed.norm() > 30)

    def use_main(self):
        self.__path.use_main()

    def on_stuck(self):
        self.__path.on_stuck()

    def on_backward(self):
        self.__path.on_backward()


def throw_projectile(context: Context, tiles_barriers):
    return (throw_washer(context) if context.is_buggy
            else throw_tire(context, tiles_barriers))


Washer = namedtuple('Washer', ('position', 'speed'))


def throw_washer(context: Context):
    washer_speed = context.game.washer_initial_speed
    washers = [
        Washer(position=context.position,
               speed=context.direction * washer_speed),
        Washer(position=context.position,
               speed=context.direction.rotate(radians(2)) * washer_speed),
        Washer(position=context.position,
               speed=context.direction.rotate(radians(-2)) * washer_speed),
    ]

    def generate():
        for car in context.opponents_cars:
            car_position = Point(car.x, car.y)
            car_speed = Point(car.speed_x, car.speed_y)
            car_barriers = list(make_units_barriers([car]))
            if car_speed.norm() < 1:
                for washer in washers:
                    yield make_has_intersection_with_lane(
                        position=washer.position,
                        course=washer.speed * 150,
                        barriers=car_barriers,
                        width=context.game.washer_radius,
                    )(0)
            else:
                car_line = Line(car_position, car_position + car_speed)
                for washer in washers:
                    washer_line = Line(washer.position,
                                       washer.position + washer.speed)
                    intersection = washer_line.intersection(car_line)
                    if intersection is None:
                        continue
                    car_dir = intersection - car_position
                    if car_dir.norm() > 0 and car_dir.cos(car_speed) < 0:
                        continue
                    if car_dir.norm() > washer_speed * 150:
                        continue
                    washer_dir = intersection - washer.position
                    if (washer_dir.norm() > 0 and
                            washer_dir.cos(washer.speed) < 0):
                        continue
                    car_time = car_dir.norm() / car_speed.norm()
                    washer_time = washer_dir.norm() / washer.speed.norm()
                    if abs(car_time - washer_time) <= WASHER_INTERVAL:
                        yield True

    return next((x for x in generate() if x), False)


def tiles_at_line(line: Line):
    tiles = bresenham(line)
    for i, current in islice(enumerate(tiles), len(tiles) - 1):
        yield current
        following = tiles[i + 1]
        if current.x != following.x and current.y != following.y:
            yield current + Point(following.x - current.x, 0)
            yield current + Point(0, following.y - current.y)


def bresenham(line: Line):
    x1 = line.begin.x
    y1 = line.begin.y
    x2 = line.end.x
    y2 = line.end.y
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1
    dy = y2 - y1
    error = int(dx / 2.0)
    y_step = 1 if y1 < y2 else -1
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = Point(y, x) if is_steep else Point(x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:
        points.reverse()
    return points


def throw_tire(context: Context, tiles_barriers):
    tire_speed = context.direction * context.game.tire_initial_speed
    width = context.world.width
    height = context.world.height

    def generate_barriers(line: Line):
        def impl():
            for tile in tiles_at_line(line):
                if 0 <= tile.x < width and 0 <= tile.y < height:
                    yield tiles_barriers[get_point_index(tile, height)]
        return chain.from_iterable(impl())

    tile_size = context.game.track_tile_size

    def has_intersection_with_tiles(distance):
        course = tire_speed.normalized() * distance
        line = Line(begin=get_current_tile(context.position, tile_size),
                    end=get_current_tile(context.position + course, tile_size))
        return make_has_intersection_with_lane(
            position=context.position,
            course=course,
            barriers=list(chain(generate_barriers(line))),
            width=context.game.tire_radius,
        )(0)

    def generate():
        for car in context.opponents_cars:
            car_position = Point(car.x, car.y)
            car_speed = Point(car.speed_x, car.speed_y)
            car_barriers = list(make_units_barriers([car]))
            distance = (context.position - car_position).norm()
            if car_speed.norm() < 1:
                yield (not has_intersection_with_tiles(distance) and
                       make_has_intersection_with_lane(
                           position=context.position,
                           course=tire_speed * 50,
                           barriers=car_barriers,
                           width=context.game.tire_radius,
                       )(0))
            else:
                car_line = Line(car_position, car_position + car_speed)
                tire_line = Line(context.position,
                                 context.position + tire_speed)
                intersection = tire_line.intersection(car_line)
                if intersection is None:
                    continue
                car_dir = intersection - car_position
                if car_dir.norm() > 0 and car_dir.cos(car_speed) < 0:
                    continue
                if car_dir.norm() > context.game.tire_initial_speed * 50:
                    continue
                tire_dir = intersection - context.position
                if tire_dir.norm() > 0 and tire_dir.cos(tire_speed) < 0:
                    continue
                if has_intersection_with_tiles(tire_dir.norm()):
                    continue
                car_time = car_dir.norm() / car_speed.norm()
                tire_time = tire_dir.norm() / tire_speed.norm()
                if abs(car_time - tire_time) <= TIRE_INTERVAL:
                    yield True

    return next((x for x in generate() if x), False)


class Path:
    def __init__(self, start_tile, get_direction, waypoints_count):
        self.__path = []
        self.__forward = ForwardWaypointsPathBuilder(
            start_tile=start_tile,
            waypoints_count=waypoints_count,
        )
        self.__backward = BackwardWaypointsPathBuilder(
            start_tile=start_tile,
            get_direction=get_direction,
            waypoints_count=waypoints_count,
        )
        self.__unstuck_backward = UnstuckPathBuilder(-1)
        self.__unstuck_forward = UnstuckPathBuilder(1)
        self.__u_turn = UTurnPathBuilder()
        self.__main = self.__forward
        self.__stuck_states = {
            id(self.__forward): self.__unstuck_backward,
            id(self.__backward): self.__unstuck_forward,
            id(self.__u_turn): self.__forward,
            id(self.__unstuck_backward): self.__unstuck_forward,
            id(self.__unstuck_forward): self.__unstuck_backward,
        }
        self.__current = self.__main
        self.__get_direction = get_direction
        self.__unknown_count = 0

    def get(self, context: Context):
        self.__update(context)
        return list(adjust_for_bonuses(
            path=self.__path,
            bonuses=context.world.bonuses,
            tile_size=context.game.track_tile_size,
            world_height=context.world.height,
            priority_conf=PriorityConf(
                durability=context.me.durability,
                projectile_left=1,
                oil_canister_left=(MAX_CANISTER_COUNT -
                                   context.me.oil_canister_count),
            ),
            limit=PATH_SIZE_FOR_BONUSES,
        ))

    def use_main(self):
        self.__current = self.__main
        self.__path.clear()

    def on_stuck(self):
        self.__current = self.__stuck_states[id(self.__current)]
        self.__path.clear()

    def on_backward(self):
        self.__current = self.__u_turn
        self.__path.clear()

    @property
    def is_main(self):
        return self.__current == self.__main

    def __update(self, context: Context):
        def need_take_next(path):
            if not path:
                return False
            tile = context.tile
            first_tile = get_current_tile(path[0], context.game.track_tile_size)
            if tile != first_tile and (tile.x == first_tile.x or
                                       tile.y == first_tile.y):
                return False
            course = path[0] - context.position
            distance = course.norm()
            if distance < min(context.me.width, context.me.height):
                return True
            return (distance < 0.75 * context.game.track_tile_size and
                    self.__get_direction().cos(course) < 0.25)

        path_size = len(self.__path)
        while need_take_next(self.__path):
            self.__path = self.__path[1:]

        def need_remake(path):
            if not path:
                return True
            tile = context.tile
            first_tile = get_current_tile(path[0], context.game.track_tile_size)
            return (
                tile.x != first_tile.x and tile.y != first_tile.y or
                path_size != len(self.__path) and
                context.speed.norm() > 0 and
                context.direction.cos(context.speed) < -cos(1) or
                path_has_tiles(path, context.world.tiles_x_y,
                               context.game.track_tile_size,
                               TileType.EMPTY) or
                path_count_tiles(path, context.world.tiles_x_y,
                                 context.game.track_tile_size,
                                 TileType.UNKNOWN) != self.__unknown_count or
                context.position.distance(path[0]) >
                2 * context.game.track_tile_size)

        if need_remake(self.__path):
            self.__path = self.__current.make(context)
            self.__unknown_count = path_count_tiles(
                self.__path, context.world.tiles_x_y,
                context.game.track_tile_size, TileType.UNKNOWN)
        self.__backward.start_tile = context.tile
        self.__forward.start_tile = context.tile


def path_count_tiles(path, tiles, tile_size, tile_type):
    return sum(path_filter_tiles(path, tiles, tile_size, tile_type))


def path_has_tiles(path, tiles, tile_size, tile_type):
    return next(path_filter_tiles(path, tiles, tile_size, tile_type), False)


def path_filter_tiles(path, tiles, tile_size, tile_type):
    for p in path:
        tile = get_current_tile(p, tile_size)
        if (0 <= tile.x < len(tiles) and 0 <= tile.y < len(tiles[tile.x]) and
                tiles[tile.x][tile.y] == tile_type):
            yield True


class WaypointsPathBuilder:
    def __init__(self, start_tile):
        self.start_tile = start_tile

    def make(self, context: Context):
        waypoints = self._waypoints(context.me.next_waypoint_index,
                                    context.world.waypoints)
        path = list(make_tiles_path(
            start_tile=context.tile,
            waypoints=waypoints,
            tiles=context.world.tiles_x_y,
            direction=context.direction,
        ))
        if not path:
            path = [self.start_tile]
        elif self.start_tile != path[0]:
            path = [self.start_tile] + path
        path = [(x + Point(0.5, 0.5)) * context.game.track_tile_size
                for x in path]
        shift = (context.game.track_tile_size / 2 -
                 context.game.track_tile_margin -
                 max(context.me.width, context.me.height) / 2)
        path = list(adjust_path(path, shift, context.game.track_tile_size))
        path = list(shift_on_direct(path))
        return path

    def _waypoints(self, next_waypoint_index, waypoints):
        raise NotImplementedError()

    def _direction(self, context: Context):
        raise NotImplementedError()


class ForwardWaypointsPathBuilder(WaypointsPathBuilder):
    def __init__(self, start_tile, waypoints_count):
        super().__init__(start_tile)
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

    def _direction(self, context: Context):
        return context.direction


class BackwardWaypointsPathBuilder(WaypointsPathBuilder):
    def __init__(self, start_tile, get_direction, waypoints_count):
        super().__init__(start_tile)
        self.__waypoints_count = waypoints_count
        self.__begin = None
        self.__get_direction = get_direction

    def make(self, context: Context):
        result = super().make(context)[1:]
        if context.speed.norm() < 1:
            first = (context.position -
                     self.__get_direction().normalized() *
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

    def _direction(self, context: Context):
        result = self.__get_direction().normalized()
        if context.speed.norm() > 0:
            return result + context.speed.normalized()
        return result


class UnstuckPathBuilder:
    def __init__(self, factor):
        self.__factor = factor

    def make(self, context: Context):
        line = Line(begin=context.position,
                    end=(context.position + context.direction * self.__factor *
                         context.game.track_tile_size))
        clipped = context.world_rectangle.clip_line(line)
        if clipped != line:
            return [clipped.begin + (line.end - line.begin) * 0.99]
        else:
            return [line.end]


class UTurnPathBuilder:
    @staticmethod
    def make(context: Context):
        tile_size = context.game.track_tile_size
        center = get_tile_center(context.tile, tile_size)
        rectangle = Rectangle(left_top=center - Point(tile_size, tile_size),
                              right_bottom=center + Point(tile_size, tile_size))
        orthogonal = context.direction.left_orthogonal()
        line = Line(begin=context.position + orthogonal * tile_size,
                    end=context.position - orthogonal * tile_size)
        line = rectangle.clip_line(line)
        farthest = max(line.begin, line.end,
                       key=lambda x: x.distance(context.position))
        return [context.position +
                (farthest - context.position).normalized() * tile_size]


def generate_cos(path):
    for i, current in islice(enumerate(path), 1, len(path) - 1):
        a = current - path[i - 1]
        b = path[i + 1] - current
        yield (1 if a.norm() < 1e-3 or b.norm() < 1e-3 else a.cos(b))


class Course:
    def __init__(self):
        self.__tile_barriers = None
        self.__tiles = None

    @property
    def tile_barriers(self):
        return self.__tile_barriers

    def get(self, context: Context, path):
        if (self.__tiles is None or self.__tile_barriers is None or
                self.__tiles != context.world.tiles_x_y):
            self.__tiles = copy(context.world.tiles_x_y)
            self.__tile_barriers = make_tiles_barriers(
                tiles=context.world.tiles_x_y,
                margin=context.game.track_tile_margin,
                size=context.game.track_tile_size,
            )
        tile_size = context.game.track_tile_size
        sub_path = [context.position] + path
        if reduce(mul, generate_cos(path), 1) < 0:
            target_position = Curve(sub_path).at(tile_size)
        else:
            target_position = max(
                Polyline(sub_path).at(tile_size),
                path[0],
                key=lambda x: x.distance(context.position)
            )
        course = target_position - context.position
        current_tile = context.tile
        target_tile = get_current_tile(target_position, tile_size)
        target_tile.x = max(0, min(context.world.width - 1, target_tile.x))
        target_tile.y = max(0, min(context.world.height - 1, target_tile.y))
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
        row_size = context.world.height

        def generate_tiles_barriers():
            def impl():
                for tile in tiles:
                    yield self.__tile_barriers[get_point_index(tile, row_size)]
            return chain.from_iterable(impl())

        tiles_barriers = list(generate_tiles_barriers())
        all_barriers = list(chain(tiles_barriers,
                                  generate_units_barriers(context)))
        width = max(context.me.width, context.me.height)
        angle = course.rotation(context.direction)

        def with_lane(current_barriers):
            return make_has_intersection_with_lane(
                position=context.position,
                course=course * (0.75 + context.speed.norm() / MAX_SPEED),
                barriers=current_barriers,
                width=width,
            )

        def adjust_forward(has_intersection, begin, end):
            return adjust_course_forward(has_intersection, angle, begin, end)

        variants = [
            lambda: adjust_forward(with_lane(all_barriers), -pi / 4, pi / 4),
            lambda: adjust_forward(with_lane(tiles_barriers), -1, 1),
        ]
        for f in variants:
            rotation = f()
            if rotation is not None:
                return course.rotate(rotation)
        return course


def generate_units_barriers(context: Context):
    return chain(
        generate_projectiles_barriers(context),
        generate_oil_slicks_barriers(context),
        generate_cars_barriers(context),
    )


def generate_projectiles_barriers(context: Context):
    return make_units_barriers(context.world.projectiles)


def generate_oil_slicks_barriers(context: Context):
    return make_units_barriers(context.world.oil_slicks)


def generate_cars_barriers(context: Context):
    def use(car: Car):
        car_speed = Point(car.speed_x, car.speed_y)
        return (car.id != context.me.id and
                (context.speed.norm() > CAR_SPEED_FACTOR * car_speed.norm() or
                 context.speed.norm() > 0 and
                 (car_speed.norm() > 0 and
                  abs(context.speed.cos(car_speed)) < cos(1) or
                  car_speed.norm() == 0)))

    return make_units_barriers(filter(use, context.world.cars))


def generate_opponents_cars_barriers(context: Context):
    return make_units_barriers(context.opponents_cars)


def adjust_course_forward(has_intersection, angle, begin, end):
    return adjust_course_rotation(has_intersection, begin, end, angle)


def adjust_course_rotation(has_intersection, begin, end, angle):
    middle = (begin + end) / 2
    if not has_intersection(middle):
        return middle
    left, right = find_false(begin, end, has_intersection, 2 ** -3)
    if left is not None:
        if right is not None:
            if abs(abs(left) - abs(right)) < 1e-8:
                return (left if abs(angle - left) < abs(angle - right)
                        else right)
            else:
                return min(left, right, key=abs)
        else:
            return left
    else:
        return right


def find_false(begin, end, is_true, min_interval):
    center = (begin + end) / 2
    left = None
    right = None
    queue = deque([(begin, end)])
    while queue:
        begin, end = queue.popleft()
        if end - begin < min_interval:
            break
        middle = (begin + end) / 2
        if is_true(middle):
            queue.append((begin, middle))
            queue.append((middle, end))
        else:
            if middle < center:
                if left is None or center - middle < center - left:
                    left = middle
                queue.appendleft((middle, end))
            else:
                if right is None or middle - center < right - center:
                    right = middle
                queue.appendleft((begin, middle))
    return left, right


class SpeedLoss:
    def __init__(self, history_size):
        self.__crush = LimitedSum(history_size)
        self.__speed = LimitedSum(history_size)
        self.__history_size = history_size

    def update(self, crush, speed):
        self.__crush.update(crush)
        self.__speed.update(speed if crush else 0)

    def get(self):
        return self.__speed.get() * self.__crush.get() / self.__crush.count

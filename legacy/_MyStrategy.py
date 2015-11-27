from collections import namedtuple
from itertools import chain
from numpy import array, meshgrid, linspace, vectorize, arctan2, sign, arange, dot
from scipy.sparse.csgraph import dijkstra
from scipy.interpolate import UnivariateSpline
from math import sqrt, cos, sin, pi, log, exp, acos, tan
from matplotlib.pyplot import show, ion, figure
from mpl_toolkits.mplot3d import Axes3D
from itertools import islice, takewhile
from scipy.optimize import fminbound, bisect
from enum import Enum

from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
from model.TileType import TileType
from model.RectangularUnit import RectangularUnit
from model.CircularUnit import CircularUnit


class MyStrategy:
    controller = None
    path = None
    target = None

    def __init__(self):
        self.plot = Plot()
        ion()
        show()

    def inc_target(self):
        self.target = (self.target + 1) % len(self.path)

    def move(self, me: Car, world: World, game: Game, move: Move):
        if self.controller is None:
            self.controller = Controller(
                distance_to_wheels=me.width / 3,
                max_engine_power_derivative=game.car_engine_power_change_per_tick,
                angular_speed_factor=game.car_angular_speed_factor)
        if world.tick < game.initial_freeze_duration_ticks:
            return
        move.spill_oil = True
        move.throw_projectile = True
        position = Point(me.x, me.y)
        my_speed = Point(me.speed_x, me.speed_y)
        my_direction = Point(1, 0).rotate(me.angle)
        # print(
        #     'move',
        #     'tick:', world.tick,
        #     'position:', me.x, me.y,
        #     'angle:', me.angle,
        #     'speed:', me.speed_x, me.speed_y, my_speed.norm(),
        #     'angular speed:', me.angular_speed
        #     'engine power:', me.engine_power,
        #     'wheel turn:', me.wheel_turn,
        # )
        matrix = AdjacencyMatrix(world.tiles_x_y)
        tile = current_tile(Point(me.x, me.y), game.track_tile_size)
        tile_index = matrix.index(tile.x, tile.y)
        path = list(make_path(tile_index, me.next_waypoint_index, matrix,
                              world.waypoints + [(tile.x, tile.y)]))
        path = [tile_center(x, game.track_tile_size) for x in path]
        # tile_center_path = path
        path = list(adjust_path(path, game.track_tile_size))
        # shifted_path = list(shift_to_borders(path))
        # shifted_polar_path = list(polar(my_position, [my_position] + path))
        # shifted_path_for_spline = list(take_for_spline(shifted_polar_path))
        # path = list(reduce_direct(path))
        # path = list(reduce_diagonal_direct(path))
        path = list(shift_on_direct(path))
        self.path = path
        self.target = 0
        path = self.path
        target = self.target
        tile_size = game.track_tile_size
        target_distance = position.distance(path[target])
        if target_distance < 0.25 * tile_size:
            self.inc_target()
        if (target < len(path) - 1 and
                position.distance(path[target + 1]) < target_distance or
                target >= len(path) and
                position.distance(path[0]) < target_distance):
            self.inc_target()
        if target < len(path) - 1:
            target_speed = get_speed(position, path[target], path[target + 1])
        else:
            target_speed = get_speed(position, path[target], path[0])
        # path = list(reduce_direct_first_after_me(path))
        # for_spline = [(x - my_position).rotate(-me.angle) for x in path_from_me]
        # for_spline = list(take_for_spline(for_spline))
        # spline = make_spline(for_spline)
        # target_x = for_spline[-1].x / 2
        # target_y = spline(target_x)
        # target = Point(target_x, target_y).rotate(me.angle) + my_position
        # path_curve = [Point(x, spline(x)).rotate(me.angle) + my_position
        #               for x in linspace(0, for_spline[-1].x, 100)]
        control = self.controller(
            position=position,
            angle=me.angle,
            angle_error=me.get_angle_to(path[target].x, path[target].y),
            wheel_turn=me.wheel_turn,
            engine_power=me.engine_power,
            speed=my_speed,
            angular_speed=me.angular_speed,
            target_position=path[target],
            speed_at_target=target_speed,
            tick=world.tick
        )
        move.engine_power += control.engine_power_derivative
        move.wheel_turn += control.wheel_turn_derivative
        # error = Polyline(path_from_me).distance(my_position)
        # error = array([
        #     reduced_path[0].distance(my_position),
        #     50 - my_speed.norm(),
        #     0.0 if my_speed.norm() == 0.0
        #     else 1.0 - (reduced_path[0] - my_position).cos(my_speed)
        # ])
        # output = self.controller(error)
        # move.engine_power = output[1]
        # move.wheel_turn = output[2]
        # self.engine_power_history.append(move.engine_power)
        # polar_path = list(polar(my_position, path_from_me))
        # path_for_spline = list(take_for_spline(polar_path))
        # my_radius = min((me.height, me.width)) / 2
        # my_speed = Point(me.speed_x, me.speed_y)
        # barriers = []
        # tiles = []
        # for position in islice(path, len(shifted_path_for_spline)):
        #     barriers += make_tile_barriers(
        #         world.tiles_x_y[position.x][position.y], position,
        #         game.track_tile_margin, game.track_tile_size)
        #     tiles.append(position)
        # barriers += make_units_barriers((c for c in world.cars
        #                                  if c.id != me.id))
        # barriers += make_units_barriers(world.projectiles)
        # passability = make_passability_function(barriers, my_radius, my_speed,
        #                                         tiles, game.track_tile_size)

        # def polar_passability(radius, angle):
        #     cartesian = Point(radius, angle).cartesian(my_position)
        #     return passability(cartesian.x, cartesian.y)

        # trajectory_points = list(make_trajectory(passability, path_for_spline,
        #                                          my_position))
        #
        # move.engine_power = 0.5
        # if world.tick % 50 == 0:
            # trajectory_spline = make_spline(trajectory_points)
            # trajectory_points = [p.cartesian(my_position)
            #                      for p in trajectory_points]
            # trajectory_spline_points = [
            #     Point(r, trajectory_spline(r))
            #     for r in linspace(0, path_for_spline[-1].radius, 100)]
            # trajectory_spline_points = [
            #     p.cartesian(my_position) for p in trajectory_spline_points]
            # self.plot.clear()
            # self.plot.path([Point(p.x, -p.y) for p in tile_center_path], 'o')
            # self.plot.path([Point(p.x, -p.y) for p in tile_center_path], '-')
            # self.plot.path([Point(p.x, -p.y) for p in adjusted_path], 'o')
            # self.plot.path([Point(p.x, -p.y) for p in adjusted_path], '-')
            # self.plot.path([Point(p.x, -p.y) for p in path], 'o')
            # self.plot.path([Point(p.x, -p.y) for p in path], '-')
            # self.plot.path(path_curve, '-')
            # self.plot.path([target], 'o')
            # self.plot.draw()
            # self.plot.path(trajectory_points, 'o')
            # self.plot.path(trajectory_points, '-')
            # self.plot.path(trajectory_spline_points, '-')
            # self.plot.surface(
            #     linspace(0, world.width * game.track_tile_size, 150),
            #     linspace(world.height * game.track_tile_size, 0, 150),
            #     passability)


# PathPoint = namedtuple('PathPoint', ('position', 'speed'))

TypedPoint = namedtuple('TypedPoint', ('position', 'type'))

# def make_trajectory(passability, path_points, origin):
#     yield path_points[0]
#     previous = path_points[0].cartesian(origin)
#     target = path_points[1].cartesian(origin)
#     target_index = 1
#     last_direction = target - previous
#     previous_angle = (last_direction / last_direction.norm()).polar(origin).angle
#     radius_iter = islice(arange(100, path_points[-1].radius - 100, 100), 30)
#     for radius in radius_iter:
#         def func(a):
#             cartesian = Point(radius, a).cartesian(origin)
#             direction = cartesian - previous
#             distance = cartesian.distance(target)
#             return (0.0
#                 - direction.cos(last_direction)
#                 + 2 * (1 - passability(cartesian.x, cartesian.y))
#                 + distance / 100
#             )
#         angle = fminbound(func, previous_angle - pi, previous_angle + pi)
#         point = Point(radius, angle)
#         yield point
#         previous_angle = angle
#         current = point.cartesian(origin)
#         if current.distance(target) < 500:
#             target_index += 1
#             target = path_points[target_index]
#             last_direction = target - current
#         else:
#             last_direction = current - previous
#         previous = current


def make_tile_rectangle(position, size):
    center = tile_center(position, size)
    to_corner = Point(size / 2, size / 2)
    return Rectangle(left_top=center - to_corner,
                     right_bottom=center + to_corner)


def make_spline(path):
    path_x = array([p.x for p in path])
    path_y = array([p.y for p in path])
    return UnivariateSpline(path_x, path_y, k=min(len(path_x) - 1, 5))


Node = namedtuple('Node', ('x', 'y'))

# def graph_plot(matrix):
#     x_max = max(matrix.x_position(x) for x in range(len(matrix.values)))
#     y_max = max(matrix.y_position(x) for x in range(len(matrix.values)))
#     pyplot.figure()
#     for x, v in enumerate(matrix.values):
#         s = array([matrix.x_position(x), matrix.y_position(x)])
#         pyplot.plot([s[0]], [s[1]], 'o')
#         for y, w in enumerate(v):
#             if w:
#                 d = array([matrix.x_position(y), matrix.y_position(y)]) - s
#                 pyplot.arrow(s[0], s[1], d[0], d[1], head_width=0.2, head_length=0.2, fc='k', ec='k')
#     pyplot.axis([-1, x_max + 1, -1, y_max + 1])
#     pyplot.show()


PointWithSpeed = namedtuple('PointWithSpeed', ('position', 'speed'))
MAX_SPEED = 2


def get_speed(position, following, after_following):
    direction = (after_following - following).normalized()
    to_following = following - position
    to_after_following = after_following - following
    return (direction * get_speed_gain(to_following.cos(to_after_following)) +
            to_following / 300)


def get_speed_gain(x):
    return - MAX_SPEED / (x - 1)


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


def reduce_direct(path):
    return reduce_base_on_three(path, is_direct)


def is_direct(previous, current, following):
    return (current.x == previous.x and current.x == following.x or
            current.y == previous.y and current.y == following.y)


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


def reduce_base_on_three(path, need_reduce):
    if not path:
        return []
    yield path[0]
    if len(path) == 1:
        return
    for i, current in islice(enumerate(path), 1, len(path) - 1):
        if not need_reduce(path[i - 1], current, path[i + 1]):
            yield current
    yield path[-1]


def is_diagonal_direct(previous, current, following):
    to_previous = previous - current
    to_following = following - current
    return to_following.x == -to_previous.x and to_following.y == -to_previous.y


def tile_center(point, size):
    return point.map(lambda x: tile_center_coord(x, size))


def tile_center_coord(value, size):
    return (value + 0.5) * size


def make_tile_barriers(tile_type: TileType, position: Point, margin, size):
    absolute_position = position * size

    def point(x, y):
        return absolute_position + Point(x, y)

    left = Rectangle(left_top=point(0, 0), right_bottom=point(margin, size))
    right = Rectangle(left_top=point(size - margin, 0),
                      right_bottom=point(size, size))
    top = Rectangle(left_top=point(0, 0), right_bottom=point(size, margin))
    bottom = Rectangle(left_top=point(0, size - margin),
                       right_bottom=point(size, size))
    left_top = Circle(point(0, 0), margin)
    left_bottom = Circle(point(0, size), margin)
    right_top = Circle(point(size, 0), margin)
    right_bottom = Circle(point(size, size), margin)
    if tile_type == TileType.VERTICAL:
        return [left, right]
    elif tile_type == TileType.HORIZONTAL:
        return [top, bottom]
    elif tile_type == TileType.LEFT_TOP_CORNER:
        return [left, top, right_bottom]
    elif tile_type == TileType.RIGHT_TOP_CORNER:
        return [right, top, left_bottom]
    elif tile_type == TileType.LEFT_BOTTOM_CORNER:
        return [left, bottom, right_top]
    elif tile_type == TileType.RIGHT_BOTTOM_CORNER:
        return [right, bottom, left_top]
    elif tile_type == TileType.LEFT_HEADED_T:
        return [left_top, left_bottom, right]
    elif tile_type == TileType.RIGHT_HEADED_T:
        return [right_top, right_bottom, left]
    elif tile_type == TileType.TOP_HEADED_T:
        return [left_top, right_top, bottom]
    elif tile_type == TileType.BOTTOM_HEADED_T:
        return [left_bottom, right_bottom, top]
    elif tile_type == TileType.CROSSROADS:
        return [left_top, left_bottom, right_top, right_bottom]
    else:
        return []


def make_units_barriers(units):
    return [unit_barriers(x) for x in units]


def unit_barriers(unit):
    if isinstance(unit, RectangularUnit):
        radius = min((unit.height, unit.width)) / 2
    elif isinstance(unit, CircularUnit):
        radius = unit.radius
    else:
        radius = 1.0
    return Unit(position=Point(unit.x, unit.y), radius=radius,
                speed=Point(unit.speed_x, unit.speed_y))


def world_passability_function(tiles_functions, tile_size):
    def impl(x, y):
        tile = current_tile(Point(x, y), tile_size)
        return tiles_functions[tile.x][tile.y](x, y)
    return impl


def make_passability_function(barriers, radius, speed, tiles, tile_size):
    def impl(x, y):
        tile = current_tile(Point(x, y), tile_size)
        if tile not in tiles:
            return 0.0
        return min((b.passability(Point(x, y), radius, speed)
                    for b in barriers), default=1.0)
    return impl


Control = namedtuple('Control', ('engine_power_derivative',
                                 'wheel_turn_derivative',
                                 'brake'))


def limit(value):
    return max(-1.0, min(1.0, value))

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


def adjust_path(path, tile_size):
    if len(path) < 2:
        return (x for x in path)

    def generate():
        typed_path = list(make_typed_path())
        for i, p in islice(enumerate(typed_path), len(typed_path) - 1):
            yield adjust_path_point(p, typed_path[i + 1], tile_size)
        yield path[-1]

    def make_typed_path():
        yield TypedPoint(path[0],
                         PointType(SideType.UNKNOWN,
                                   get_point_output_type(path[0], path[1])))
        for i, p in islice(enumerate(path), 1, len(path) - 1):
            yield TypedPoint(p, get_point_type(path[i - 1], p, path[i + 1]))
        yield TypedPoint(path[-1],
                         PointType(get_point_input_type(path[-2], path[-1]),
                                   SideType.UNKNOWN))

    return generate()


def adjust_path_point(current: TypedPoint, following: TypedPoint, tile_size):
    if current.type == following.type:
        return current.position
    elif current.type in {PointType.LEFT_TOP, PointType.TOP_LEFT}:
        return current.position + Point(- tile_size / 4, - tile_size / 4)
    elif current.type in {PointType.LEFT_BOTTOM, PointType.BOTTOM_LEFT}:
        return current.position + Point(- tile_size / 4, + tile_size / 4)
    elif current.type in {PointType.RIGHT_TOP, PointType.TOP_RIGHT}:
        return current.position + Point(+ tile_size / 4, - tile_size / 4)
    elif current.type in {PointType.RIGHT_BOTTOM, PointType.BOTTOM_RIGHT}:
        return current.position + Point(+ tile_size / 4, + tile_size / 4)
    elif current.type in {PointType.LEFT_RIGHT, PointType.RIGHT_LEFT}:
        if following.type.output == SideType.TOP:
            return current.position + Point(0, tile_size / 4)
        else:
            return current.position - Point(0, tile_size / 4)
    elif current.type in {PointType.TOP_BOTTOM, PointType.BOTTOM_TOP}:
        if following.type.output == SideType.LEFT:
            return current.position + Point(tile_size / 4, 0)
        else:
            return current.position - Point(tile_size / 4, 0)
    else:
        return current.position


def get_point_type(previous, current, following):
    return PointType(get_point_input_type(previous, current),
                     get_point_output_type(current, following))


def get_point_input_type(previous, current):
    if previous.y == current.y:
        if previous.x < current.x:
            return SideType.LEFT
        else:
            return SideType.RIGHT
    if previous.x == current.x:
        if previous.y < current.y:
            return SideType.TOP
        else:
            return SideType.BOTTOM
    return SideType.UNKNOWN


def get_point_output_type(current, following):
    if current.y == following.y:
        if current.x < following.x:
            return SideType.RIGHT
        else:
            return SideType.LEFT
    if current.x == following.x:
        if current.y < following.y:
            return SideType.BOTTOM
        else:
            return SideType.TOP
    return SideType.UNKNOWN


class SideType(Enum):
    UNKNOWN = 0
    LEFT = 1
    RIGHT = 2
    TOP = 3
    BOTTOM = 4


PointTypeImpl = namedtuple('PointType', ('input', 'output'))


class PointType(PointTypeImpl):
    LEFT_RIGHT = PointTypeImpl(SideType.LEFT, SideType.RIGHT)
    LEFT_TOP = PointTypeImpl(SideType.LEFT, SideType.TOP)
    LEFT_BOTTOM = PointTypeImpl(SideType.LEFT, SideType.BOTTOM)
    RIGHT_LEFT = PointTypeImpl(SideType.RIGHT, SideType.LEFT)
    RIGHT_TOP = PointTypeImpl(SideType.RIGHT, SideType.TOP)
    RIGHT_BOTTOM = PointTypeImpl(SideType.RIGHT, SideType.BOTTOM)
    TOP_LEFT = PointTypeImpl(SideType.TOP, SideType.LEFT)
    TOP_RIGHT = PointTypeImpl(SideType.TOP, SideType.RIGHT)
    TOP_BOTTOM = PointTypeImpl(SideType.TOP, SideType.BOTTOM)
    BOTTOM_LEFT = PointTypeImpl(SideType.BOTTOM, SideType.LEFT)
    BOTTOM_RIGHT = PointTypeImpl(SideType.BOTTOM, SideType.RIGHT)
    BOTTOM_TOP = PointTypeImpl(SideType.BOTTOM, SideType.TOP)


def sigmoid(x):
    return 1 / (1 + exp(-x))


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


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return 'Point(x={x}, y={y})'.format(x=self.x, y=self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return Point(self.x * other, self.y * other)

    def __truediv__(self, other):
        return Point(self.x / other, self.y / other)

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y

    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y

    def __imul__(self, other):
        self.x *= other
        self.y *= other

    def __itruediv__(self, other):
        self.x /= other
        self.y /= other

    def __neg__(self):
        return Point(-self.x, -self.y)

    @property
    def radius(self):
        return self.x

    @radius.setter
    def radius(self, value):
        self.x = value

    @property
    def angle(self):
        return self.y

    @angle.setter
    def angle(self, value):
        self.y = value

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def norm(self):
        return sqrt(self.dot(self))

    def cos(self, other):
        return self.dot(other) / (self.norm() * other.norm())

    def distance(self, other):
        return (other - self).norm()

    def map(self, function):
        return Point(function(self.x), function(self.y))

    def polar(self, cartesian_origin=None):
        if cartesian_origin:
            return (self - cartesian_origin).polar()
        else:
            radius = self.norm()
            angle = arctan2(self.y, self.x)
            return Point(radius, angle)

    def cartesian(self, cartesian_origin=None):
        if cartesian_origin:
            return self.cartesian() + cartesian_origin
        else:
            return Point(x=self.radius * cos(self.angle),
                         y=self.radius * sin(self.angle))

    def orthogonal(self):
        return Point(-self.y, self.x)

    def rotation(self, other):
        return acos(self.cos(other)) * sign(self.y * other.x - self.x * other.y)

    def rotate(self, angle):
        return Point(self.x * cos(angle) - self.y * sin(angle),
                     self.y * cos(angle) + self.x * sin(angle))

    def normalized(self):
        return self / self.norm()

    def projection(self, other):
        return other * self.dot(other) / other.norm()


class Line:
    def __init__(self, begin: Point, end: Point):
        self.begin = begin
        self.end = end

    def __call__(self, parameter):
        return self.begin + (self.end - self.begin) * parameter

    def distance(self, point):
        to_end = self.end - self.begin
        to_point = point - self.begin
        norm = to_point.dot(to_end) / to_end.norm()
        return sqrt(to_point.norm() ** 2 - norm ** 2)

    def nearest(self, point):
        to_end = self.end - self.begin
        to_point = point - self.begin
        norm = to_point.dot(to_end) / to_end.norm()
        return self.begin + to_end / to_end.norm() * norm

    def length(self):
        return (self.end - self.begin).norm()


class AdjacencyMatrix:
    def __init__(self, tiles):
        column_size = len(tiles)
        self.__row_size = len(tiles[0])

        def generate():
            for x, column in enumerate(tiles):
                for y, tile in enumerate(column):
                    yield adjacency_matrix_row(Node(x, y), tile)

        def adjacency_matrix_row(node, tile):
            if tile == TileType.VERTICAL:
                return matrix_row({top(node), bottom(node)})
            elif tile == TileType.HORIZONTAL:
                return matrix_row({left(node), right(node)})
            elif tile == TileType.LEFT_TOP_CORNER:
                return matrix_row({right(node), bottom(node)})
            elif tile == TileType.RIGHT_TOP_CORNER:
                return matrix_row({left(node), bottom(node)})
            elif tile == TileType.LEFT_BOTTOM_CORNER:
                return matrix_row({right(node), top(node)})
            elif tile == TileType.RIGHT_BOTTOM_CORNER:
                return matrix_row({left(node), top(node)})
            elif tile == TileType.LEFT_HEADED_T:
                return matrix_row({left(node), top(node), bottom(node)})
            elif tile == TileType.RIGHT_HEADED_T:
                return matrix_row({right(node), top(node), bottom(node)})
            elif tile == TileType.TOP_HEADED_T:
                return matrix_row({top(node), left(node), right(node)})
            elif tile == TileType.BOTTOM_HEADED_T:
                return matrix_row({bottom(node), left(node), right(node)})
            elif tile == TileType.CROSSROADS:
                return matrix_row({left(node), right(node),
                                   top(node), bottom(node)})
            else:
                return matrix_row({})

        def matrix_row(dst):
            return [1 if x in dst else 0
                    for x in range(self.__row_size * column_size)]

        def left(node):
            return self.index(node.x - 1, node.y)

        def right(node):
            return self.index(node.x + 1, node.y)

        def top(node):
            return self.index(node.x, node.y - 1)

        def bottom(node):
            return self.index(node.x, node.y + 1)

        self.__values = list(generate())

    def index(self, x, y):
        return x * self.__row_size + y

    def x_position(self, index):
        return int(index / self.__row_size)

    def y_position(self, index):
        return index % self.__row_size

    @property
    def values(self):
        return self.__values


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


class Circle:
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius

    def __repr__(self):
        return 'Circle(position={p}, radius={r})'.format(
            p=repr(self.position), r=repr(self.radius))

    def __eq__(self, other):
        return (self.position == other.position and
                self.radius == other.radius)

    def passability(self, position, radius, _=None):
        distance = (self.position - position).norm()
        return float(distance > self.radius + radius)

    def intersection_with_line(self, line: Line):
        nearest = line.nearest(self.position)
        distance = self.position.distance(nearest)
        if (distance > self.radius or
                (line.begin - nearest).dot(line.end - nearest) > 0):
            return []
        if self.radius == distance:
            return [nearest]

        def generate():
            to_begin = Line(nearest, line.begin)
            if to_begin.length() > 0:
                def func(parameter):
                    return (self.position.distance(to_begin(parameter)) -
                            self.radius)
                if sign(func(0)) != sign(func(1)):
                    yield to_begin(bisect(func, 0, 1))
            to_end = Line(nearest, line.end)
            if to_end.length() > 0:
                def func(parameter):
                    return (self.position.distance(to_end(parameter)) -
                            self.radius)
                if sign(func(0)) != sign(func(1)):
                    yield to_end(bisect(func, 0, 1))

        return list(generate())


class Rectangle:
    INSIDE = 0
    LEFT = 1
    RIGHT = 2
    TOP = 4
    BOTTOM = 8

    def __init__(self, left_top, right_bottom):
        self.left_top = left_top
        self.right_bottom = right_bottom

    def __repr__(self):
        return 'Rectangle(left_top={lt}, right_bottom={rb})'.format(
            lt=repr(self.left_top), rb=repr(self.right_bottom))

    def __eq__(self, other):
        return (self.left_top == other.left_top and
                self.right_bottom == other.right_bottom)

    def passability(self, position, radius, _=None):
        position_code = self.point_code(position)
        if position_code == Rectangle.INSIDE:
            return 0.0
        width = self.right_bottom.x - self.left_top.x
        height = self.right_bottom.y - self.left_top.y
        center = self.left_top + Point(width / 2, height / 2)
        direction = center - position
        border = position + direction / direction.norm() * radius
        border_code = self.point_code(border)
        return float(position_code & border_code)

    def point_code(self, point):
        result = Rectangle.INSIDE
        if point.x < self.left_top.x:
            result |= Rectangle.LEFT
        elif point.x > self.right_bottom.x:
            result |= Rectangle.RIGHT
        if point.y < self.left_top.y:
            result |= Rectangle.TOP
        elif point.y > self.right_bottom.y:
            result |= Rectangle.BOTTOM
        return result

    def left(self):
        return Line(begin=self.left_top,
                    end=self.left_top + Point(0, self.height()))

    def right(self):
        return Line(begin=self.right_bottom,
                    end=self.right_bottom - Point(0, self.height()))

    def top(self):
        return Line(begin=self.left_top + Point(self.width(), 0),
                    end=self.left_top)

    def bottom(self):
        return Line(begin=self.right_bottom - Point(self.width(), 0),
                    end=self.right_bottom)

    def width(self):
        return self.right_bottom.x - self.left_top.x

    def height(self):
        return self.right_bottom.y - self.left_top.y


class Unit:
    def __init__(self, position, radius, speed):
        self.__circle = Circle(position, radius)
        self.__position = position
        self.__radius = radius
        self.__speed = speed

    def passability(self, position, radius, speed):
        immovable = self.__circle.passability(position, radius, speed)
        if immovable == 1.0:
            return 1.0
        else:
            distance = (self.__position - position).norm()
            return (distance / (self.__radius + radius)) ** 2

    def __repr__(self):
        return 'Unit(position={p}, radius={r}, speed={s})'.format(
            p=repr(self.__position), r=repr(self.__radius),
            s=repr(self.__speed))

    def __eq__(self, other):
        return (self.__position == other.__position and
                self.__radius == other.__radius and
                self.__speed == other.__speed)


class Plot:
    def __init__(self, title=None):
        self.__figure = figure()
        self.__axis = self.__figure.add_subplot(1, 1, 1)
        self.__title = title

    def clear(self):
        self.__axis.cla()

    def surface(self, x, y, function):
        x, y = meshgrid(x, y)
        z = vectorize(function)(x, y)
        self.__axis.imshow(z, alpha=0.5,
                           extent=[x.min(), x.max(), y.min(), y.max()])

    def path(self, points, *args, **kwargs):
        x = [p.x for p in points]
        y = [p.y for p in points]
        self.__axis.plot(x, y, *args, **kwargs)

    def curve(self, x, function, *args, **kwargs):
        y = vectorize(function)(x)
        self.__axis.plot(x, y, *args, **kwargs)

    def lines(self, x, y, *args, **kwargs):
        self.__axis.plot(x, y, *args, **kwargs)

    def draw(self):
        if self.__title is not None:
            self.__axis.set_title(self.__title)
        self.__figure.canvas.draw()


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
        self.__engine_power = PidController(0.5, 0.1, 0.1)
        self.__wheel_turn = PidController(0.2, 0.1, 0.1)
        self.__previous_full_speed = Point(0, 0)
        self.engine_power_history = []
        self.target_engine_power_history = []
        self.speed_norm_history = []
        self.target_speed_norm_history = []
        self.wheel_turn_history = []
        self.target_wheel_turn_history = []
        self.engine_power_plot = Plot('engine power')
        self.speed_plot = Plot('speed')
        self.wheel_turn_plot = Plot('wheel turn')
        ion()
        show()

    def __call__(self, position, angle, angle_error, engine_power, wheel_turn,
                 speed, angular_speed, target_position, speed_at_target, tick):
        # position_error = target_position - position
        direction = Point(1, 0).rotate(angle)
        # angle_error = position_error.rotation(direction)
        # target_angle = angle + angle_error
        target_angular_speed = angle_error
        angular_speed_error = target_angular_speed - angular_speed
        target_wheel_turn = limit(angular_speed_error)
        wheel_turn_error = target_wheel_turn - wheel_turn
        wheel_turn_derivative = self.__wheel_turn(wheel_turn_error)
        target_speed = speed_at_target
        radius = -(direction * self.distance_to_wheels).rotate(pi / 2)
        angular_speed_vec = Point(-radius.y, radius.x) * angular_speed
        # target_angular_speed_vec = (Point(-radius.y, radius.x) *
        #                             target_angular_speed)
        full_speed = speed + angular_speed_vec
        # full_speed = full_speed.projection(direction)
        # target_full_speed = target_speed + self.target_angular_speed_vec
        target_full_speed = target_speed
        # target_full_speed = target_full_speed.projection(direction)
        # full_speed_error = target_full_speed - full_speed
        # full_speed_error = (full_speed_error.norm() *
        #                     full_speed_error.cos(direction))
        full_speed_error = target_full_speed.norm() - full_speed.norm()
        acceleration = (full_speed - self.__previous_full_speed).norm()
        target_acceleration = full_speed_error
        acceleration_error = target_acceleration - acceleration
        target_engine_power = limit(acceleration_error)
        engine_power_error = target_engine_power - engine_power
        engine_power_derivative = self.__engine_power(engine_power_error)
        brake = engine_power_derivative < -self.max_engine_power_derivative
        self.__previous_full_speed = full_speed
        self.speed_norm_history.append(full_speed.norm())
        self.target_speed_norm_history.append(target_full_speed.norm())
        self.wheel_turn_history.append(wheel_turn)
        self.target_wheel_turn_history.append(target_wheel_turn)
        self.engine_power_history.append(engine_power)
        self.target_engine_power_history.append(target_engine_power)
        # print(
        #     tick,
        #     'position', position,
        #     'target_position', target_position,
        #     'position_error', position_error.normalized(),
        #     'direction:', direction,
        #     'angle:', angle,
        #     'target_angle:', target_angle,
        #     'angle_error:', angle_error,
        #     'angular_speed_error:', angular_speed_error,
        #     'wheel_turn:', wheel_turn,
        #     'target_wheel_turn:', target_wheel_turn,
        #     'wheel_turn_error:', wheel_turn_error,
        #     'wheel_turn_derivative:', wheel_turn_derivative,
        #     'full_speed:', full_speed,
        #     'full_speed:', full_speed,
        #     'target_full_speed:', target_full_speed,
        #     'full_speed_error:', full_speed_error,
        #     'acceleration:', acceleration,
        #     'target_acceleration:', target_acceleration,
        #     'acceleration_error:', acceleration_error,
        #     'engine_power_error:', acceleration_error,
        #     'target_engine_power:', target_engine_power,
        #     'engine_power_error:', engine_power_error,
        #     'engine_power_derivative:', engine_power_derivative,
        #     'brake:', brake,
        # )
        if tick % 50 == 0:
            self.speed_plot.clear()
            self.speed_plot.lines(range(len(self.speed_norm_history)),
                                  self.speed_norm_history)
            self.speed_plot.lines(range(len(self.target_speed_norm_history)),
                                  self.target_speed_norm_history)
            # self.speed_plot.lines(range(len(self.speed_norm_history)),
            #                       array(self.speed_norm_history) /
            #                       array(self.target_speed_norm_history))
            # self.speed_plot.lines(range(len(self.target_speed_norm_history)),
            #                       array(self.target_speed_norm_history) /
            #                       array(self.target_speed_norm_history))
            self.speed_plot.draw()
            self.wheel_turn_plot.clear()
            # self.wheel_turn_plot.lines(range(len(self.wheel_turn_history)),
            #                            self.wheel_turn_history)
            # self.wheel_turn_plot.lines(range(len(self.target_wheel_turn_history)),
            #                            self.target_wheel_turn_history)
            self.wheel_turn_plot.lines(range(len(self.wheel_turn_history)),
                                       array(self.wheel_turn_history) /
                                       array(self.target_wheel_turn_history))
            self.wheel_turn_plot.lines(range(len(self.target_wheel_turn_history)),
                                       array(self.target_wheel_turn_history) /
                                       array(self.target_wheel_turn_history))
            self.wheel_turn_plot.draw()
            self.engine_power_plot.clear()
            # self.engine_power_plot.lines(range(len(self.engine_power_history)),
            #                              self.engine_power_history)
            # self.engine_power_plot.lines(range(len(self.target_engine_power_history)),
            #                              self.target_engine_power_history)
            self.engine_power_plot.lines(range(len(self.engine_power_history)),
                                         array(self.engine_power_history) /
                                         array(self.target_engine_power_history))
            self.engine_power_plot.lines(range(len(self.target_engine_power_history)),
                                         array(self.target_engine_power_history) /
                                         array(self.target_engine_power_history))
            self.engine_power_plot.draw()

        return Control(engine_power_derivative, wheel_turn_derivative, brake)


def limit(value):
    return max(-1.0, min(1.0, value))


class Polyline:
    def __init__(self, points):
        self.points = points

    def distance(self, point):
        points = islice(enumerate(self.points), len(self.points) - 1)
        return min(Line(p, self.points[i - 1]).nearest(point).distance(point)
                   for i, p in points)

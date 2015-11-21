from collections import namedtuple
from itertools import chain
from numpy import array, meshgrid, linspace, vectorize, arctan2
from scipy.sparse.csgraph import dijkstra
from scipy.interpolate import UnivariateSpline
from math import sqrt, cos, sin
from matplotlib.pyplot import show, ion, figure
from mpl_toolkits.mplot3d import Axes3D
from itertools import islice, takewhile

from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
from model.TileType import TileType
from model.RectangularUnit import RectangularUnit
from model.CircularUnit import CircularUnit


class MyStrategy:
    def __init__(self):
        self.__cartesian_path_plots = PathPlots('cartesian path')
        self.__polar_path_plots = PathPlots('polar path')
        self.__path_for_spline_plots = PathPlots('path for spline')
        self.__path_spline = PathSplinePlots('path spline')
        self.__tile_passability_plot = SurfacePlot('tile_passability')
        ion()
        show()

    def move(self, me: Car, world: World, game: Game, move: Move):
        matrix = AdjacencyMatrix(world.tiles_x_y)
        tile = current_tile(me.x, me.y, game.track_tile_size)
        tile_index = matrix.index(tile.x, tile.y)
        path = list(path_to_end(tile_index, me.next_waypoint_index, matrix,
                                world.waypoints))
        tile_center_path = [tile_center(x, game.track_tile_size) for x in path]
        shifted_path = list(shift_to_borders(tile_center_path))
        # reduced_path = list(reduce_direct(shifted_path))
        # reduced_path = list(reduce_diagonal_direct(reduced_path))
        # reduced_path = list(reduce_direct_first_after_me(reduced_path))
        reduced_path = shifted_path
        my_position = Point(me.x, me.y)
        path_from_me = [my_position] + reduced_path
        polar_path = list(polar(my_position, path_from_me))
        path_for_spline = list(take_for_spline(polar_path))
        path_spline = make_spline(path_for_spline)
        target_radius = Point(me.speed_x, me.speed_y).norm() * 2
        polar_target = Point(target_radius, path_spline(target_radius))
        my_radius = min((me.height, me.width)) / 2
        my_speed = Point(me.speed_x, me.speed_y)
        barriers = []
        for position in islice(path, 2):
            barriers += tile_barriers(
                world.tiles_x_y[position.x][position.y], position,
                game.track_tile_margin, game.track_tile_size)
        barriers += units_barriers((c for c in world.cars if c.id != me.id))
        barriers += units_barriers(world.projectiles)
        passability = passability_function(barriers, my_radius, my_speed)
        move.engine_power = 0.5
        print(
            'move',
            'tick:', world.tick,
            'me:', my_position.x, my_position.y,
            'tile:', tile.x, tile.y,
            'my speed:', me.speed_x, me.speed_y,
            'angular speed:', me.angular_speed,
            'wheel turn:', me.wheel_turn,
            'width:', me.width,
            'height:', me.height,
        )
        if world.tick % 50 == 0:
            self.__cartesian_path_plots.draw(path_from_me)
            self.__polar_path_plots.draw(polar_path)
            self.__path_for_spline_plots.draw(path_for_spline)
            self.__path_spline.draw(path_for_spline, path_spline)
            if path[0].x == path[1].x:
                x = linspace(path[0].x * game.track_tile_size,
                             (path[0].x + 1) * game.track_tile_size,
                             20)
            else:
                x = linspace(min((path[0].x, path[1].x)) * game.track_tile_size,
                             (max((path[0].x, path[1].x)) + 1) * game.track_tile_size,
                             40)
            if path[0].y == path[1].y:
                y = linspace(path[0].y * game.track_tile_size,
                             (path[0].y + 1) * game.track_tile_size,
                             20)
            else:
                y = linspace(min((path[0].y, path[1].y)) * game.track_tile_size,
                             (max((path[0].y, path[1].y)) + 1) * game.track_tile_size,
                             40)
            self.__tile_passability_plot.draw(x, y, passability)


def make_spline(path):
    path_x = array([p.x for p in path])
    path_y = array([p.y for p in path])
    return UnivariateSpline(path_x, path_y)


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

    def polar(self):
        radius = self.norm()
        angle = arctan2(self.y, self.x)
        return Point(radius, angle)

    def cartesian(self):
        return Point(x=self.radius * cos(self.angle),
                     y=self.radius * sin(self.angle))


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


def current_tile(x, y, tile_size):
    return Point(tile_coord(x, tile_size), tile_coord(y, tile_size))


def tile_coord(value, tile_size):
    return int(value / tile_size)


def path_to_end(start_index, next_waypoint_index, matrix, waypoints):
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
    return ((x - origin).polar() for x in path)


def take_for_spline(path):
    if not path:
        return []

    def predicate(index, current):
        return current.radius > path[index - 1].radius

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


def tile_barriers(tile_type: TileType, position: Point, margin, size):
    low = margin
    high = size - low
    absolute_position = position * size

    def point(x, y):
        return absolute_position + Point(x, y)

    left = Border(point(low, 0), point(low, size), Point(1, 0))
    right = Border(point(high, 0), point(high, size), Point(-1, 0))
    top = Border(point(0, low), point(size, low), Point(0, 1))
    bottom = Border(point(0, high), point(size, high), Point(0, -1))
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


def units_barriers(units):
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
        tile = current_tile(x, y, tile_size)
        return tiles_functions[tile.x][tile.y](x, y)
    return impl


def passability_function(barriers, radius, speed):
    def impl(x, y):
        if barriers:
            return min(b.passability(Point(x, y), radius, speed)
                       for b in barriers)
        else:
            return 0.0
    return impl


class Circle:
    def __init__(self, position, radius):
        self.__position = position
        self.__radius = radius

    def passability(self, position, radius, _=None):
        distance = (self.__position - position).norm()
        return float(distance > self.__radius + radius)

    def __repr__(self):
        return 'Circle(position={p}, radius={r})'.format(
            p=repr(self.__position), r=repr(self.__radius))

    def __eq__(self, other):
        return (self.__position == other.__position and
                self.__radius == other.__radius)


class Border:
    def __init__(self, begin, end, normal):
        self.__begin = begin
        self.__end = end
        self.__normal = normal

    def passability(self, position, radius, _=None):
        to_end = self.__end - self.__begin
        if (to_end.x != 0 and (position.x < self.__begin.x or
                               position.x > self.__end.x) or
            to_end.y != 0 and (position.y < self.__begin.y or
                               position.y > self.__end.y)):
            return 1.0
        to_car = position - self.__begin
        distance = sqrt(to_car.norm() ** 2 -
                        (to_car.dot(to_end) / to_end.norm()) ** 2)
        if distance <= radius:
            return 0.0
        return float(to_car.dot(self.__normal) > 0)

    def __repr__(self):
        return 'Border(begin={b}, end={e}, normal={n})'.format(
            b=repr(self.__begin), e=repr(self.__end),
            n=repr(self.__normal))

    def __eq__(self, other):
        return (self.__begin == other.__begin and
                self.__end == other.__end and
                self.__normal == other.__normal)


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
        return immovable

    def __repr__(self):
        return 'Unit(position={p}, radius={r}, speed={s})'.format(
            p=repr(self.__position), r=repr(self.__radius),
            s=repr(self.__speed))

    def __eq__(self, other):
        return (self.__position == other.__position and
                self.__radius == other.__radius and
                self.__speed == other.__speed)


class PathPlots:
    def __init__(self, title):
        self.__title = title
        self.__figure = figure()
        self.__axis = self.__figure.add_subplot(1, 1, 1)

    def draw(self, path):
        path_x = array([p.x for p in path])
        path_y = array([p.y for p in path])
        step_x = abs((max(path_x) - min(path_x)) / 10)
        step_y = abs((max(path_y) - min(path_y)) / 10)
        self.__axis.cla()
        self.__axis.set_title(self.__title)
        self.__axis.set_xlim([min(path_x) - step_x, max(path_x) + step_x])
        self.__axis.set_ylim([min(path_y) - step_y, max(path_y) + step_y])
        self.__axis.plot(path_x, path_y, 'o', path_x, path_y, '-')
        self.__figure.canvas.draw()


class PathSplinePlots:
    def __init__(self, title):
        self.__title = title
        self.__figure = figure()
        self.__axis = self.__figure.add_subplot(1, 1, 1)

    def draw(self, path, path_spline):
        self.__axis.cla()
        self.__axis.set_title(self.__title)
        path_x = array([p.x for p in path])
        path_y = array([p.y for p in path])
        self.__axis.plot(path_x, path_y, 'o', path_x, path_y, '-')
        spline_x = linspace(0, path[-1].x, 100)
        self.__axis.plot(spline_x, vectorize(path_spline)(spline_x))
        self.__figure.canvas.draw()


class SurfacePlot:
    def __init__(self, title):
        self.__title = title
        self.__figure = figure()
        self.__axis = self.__figure.add_subplot(1, 1, 1)

    def draw(self, x, y, function):
        x, y = meshgrid(x, y)
        z = vectorize(function)(x, y)
        self.__axis.cla()
        self.__axis.set_title(self.__title)
        self.__axis.imshow(z)
        self.__figure.canvas.draw()

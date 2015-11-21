from collections import namedtuple
from itertools import chain
from numpy import array, meshgrid, linspace, vectorize, arctan2
from scipy.sparse.csgraph import dijkstra
from math import sqrt, cos, sin
from matplotlib.pyplot import show, ion, figure
from mpl_toolkits.mplot3d import Axes3D
from itertools import islice, takewhile

from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
from model.TileType import TileType


class MyStrategy:
    def __init__(self):
        # self.__path_plots = PathPlots('path')
        # self.__tile_center_path_plots = PathPlots('tile_center_path')
        # self.__shifted_path_plots = PathPlots('shifted_path')
        self.__cartesian_path_plots = PathPlots('cartesian path')
        self.__polar_path_plots = PathPlots('polar path')
        self.__path_for_spline_plots = PathPlots('path for spline')
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
        reduced_path = list(reduce_direct(shifted_path))
        reduced_path = list(reduce_diagonal_direct(reduced_path))
        reduced_path = list(reduce_direct_first_after_me(reduced_path))
        me_position = Point(me.x, me.y)
        path_from_me = [me_position] + reduced_path
        polar_path = list(polar(me_position, path_from_me))
        path_for_spline = list(take_for_spline(polar_path))
        barriers = tile_barriers(world.tiles_x_y[tile.x][tile.y],
                                 game.track_tile_margin, game.track_tile_size)
        passability = tile_passability(barriers, me.height, me.width)
        move.engine_power = 0.5
        print(
            'move',
            'tick:', world.tick,
            'me: ', me_position,
            'me_polar: ', me_position.polar(),
            'tile: ', tile,
        )
        if world.tick % 100 == 0:
            self.__cartesian_path_plots.draw(path_from_me)
            self.__polar_path_plots.draw(polar_path)
            self.__path_for_spline_plots.draw(path_for_spline)
            self.__tile_passability_plot.draw(
                linspace(0, game.track_tile_size, 20), passability)


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


def current_tile(x, y, track_tile_size):
    return Point(tile_coord(x, track_tile_size), tile_coord(y, track_tile_size))


def tile_coord(value, track_tile_size):
    return int(value / track_tile_size)


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


def tile_barriers(tile_type, tile_margin, tile_size):
    low = tile_margin
    high = tile_size - low
    left = Border(Point(low, 0), Point(low, tile_size), Point(1, 0))
    right = Border(Point(high, 0), Point(high, tile_size), Point(-1, 0))
    top = Border(Point(0, low), Point(tile_size, low), Point(0, 1))
    bottom = Border(Point(0, high), Point(tile_size, high), Point(0, -1))
    left_top = Circle(Point(0, 0), tile_margin)
    left_bottom = Circle(Point(0, tile_size), tile_margin)
    right_top = Circle(Point(tile_size, 0), tile_margin)
    right_bottom = Circle(Point(tile_size, tile_size), tile_margin)
    if tile_type == TileType.VERTICAL:
        return left, right
    elif tile_type == TileType.HORIZONTAL:
        return top, bottom
    elif tile_type == TileType.LEFT_TOP_CORNER:
        return left, top, right_bottom
    elif tile_type == TileType.RIGHT_TOP_CORNER:
        return right, top, left_bottom
    elif tile_type == TileType.LEFT_BOTTOM_CORNER:
        return left, bottom, right_top
    elif tile_type == TileType.RIGHT_BOTTOM_CORNER:
        return right, bottom, left_top
    elif tile_type == TileType.LEFT_HEADED_T:
        return left_top, left_bottom, right
    elif tile_type == TileType.RIGHT_HEADED_T:
        return right_top, right_bottom, left
    elif tile_type == TileType.TOP_HEADED_T:
        return left_top, right_top, bottom
    elif tile_type == TileType.BOTTOM_HEADED_T:
        return left_bottom, right_bottom, top
    elif tile_type == TileType.CROSSROADS:
        return left_top, left_bottom, right_top, right_bottom
    else:
        return tuple()


def tile_passability(barriers, height, width):
    def impl(x, y):
        return min(b.passability(x, y, height, width) for b in barriers)
    return impl


class Barrier:
    def passability(self, x, y, height, width):
        raise NotImplementedError()


class Circle(Barrier):
    def __init__(self, position, radius):
        self.__position = position
        self.__radius = radius

    def passability(self, x, y, height, width):
        distance = (self.__position - Point(x, y)).norm()
        radius = max((height, width)) / 2
        return float(distance > self.__radius + radius)

    def __repr__(self):
        return 'Barrier(position={p}, radius={r})'.format(
            p=repr(self.__position), r=repr(self.__radius))

    def __eq__(self, other):
        return (self.__position == other.__position and
                self.__radius == other.__radius)


class Border(Barrier):
    def __init__(self, begin, end, normal):
        self.__begin = begin
        self.__end = end
        self.__normal = normal

    def passability(self, x, y, height, width):
        position = Point(x, y)
        to_car = position - self.__begin
        to_end = self.__end - self.__begin
        distance = sqrt(to_car.norm() ** 2 -
                        (to_car.dot(to_end) / to_end.norm()) ** 2)
        if distance <= max((height, width)) / 2:
            return 0.0
        return float(to_car.dot(self.__normal) > 0)

    def __repr__(self):
        return 'Barrier(begin={b}, end={e}, normal={n})'.format(
            b=repr(self.__begin), e=repr(self.__end), n=repr(self.__normal))

    def __eq__(self, other):
        return (self.__begin == other.__begin and
                self.__end == other.__end and
                self.__normal == other.__normal)


class PathPlots:
    def __init__(self, title):
        self.__figure = figure()
        self.__axis = self.__figure.add_subplot(1, 1, 1)
        self.__axis.set_title(title)

    def draw(self, path):
        path_x = array([p.x for p in path])
        path_y = array([p.y for p in path])
        step_x = abs((max(path_x) - min(path_x)) / 10)
        step_y = abs((max(path_y) - min(path_y)) / 10)
        self.__axis.cla()
        self.__axis.set_xlim([min(path_x) - step_x, max(path_x) + step_x])
        self.__axis.set_ylim([min(path_y) - step_y, max(path_y) + step_y])
        self.__axis.plot(path_x, path_y, 'o', path_x, path_y, '-')
        self.__figure.canvas.draw()


class SurfacePlot:
    def __init__(self, title):
        self.__figure = figure()
        self.__axis = self.__figure.add_subplot(1, 1, 1, projection='3d')
        self.__axis.set_title(title)

    def draw(self, limits, function):
        x, y = meshgrid(limits, limits)
        z = vectorize(function)(x, y)
        self.__axis.cla()
        self.__axis.plot_wireframe(x, y, z)
        self.__figure.canvas.draw()

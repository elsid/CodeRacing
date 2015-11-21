from collections import namedtuple
from itertools import chain
from numpy import array, dot, array_equal, meshgrid, linspace, vectorize
from numpy.linalg import norm
from scipy.sparse.csgraph import dijkstra
from math import sqrt
from matplotlib.pyplot import subplots, show, ion, figure
from mpl_toolkits.mplot3d import Axes3D

from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
from model.TileType import TileType


class MyStrategy:
    def __init__(self):
        self.__path_plots = PathPlots()
        self.__detailed_path_plots = PathPlots()
        self.__tiles_barriers_plot = SurfacePlot()
        ion()
        show()

    def move(self, me: Car, world: World, game: Game, move: Move):
        print('move', world.tick, me.x, me.y)
        matrix = AdjacencyMatrix(world.tiles_x_y)
        tile = current_tile(me.x, me.y, game.track_tile_size)
        tile_index = matrix.index(tile.x, tile.y)
        path = list(path_to_end(tile_index, me.next_waypoint_index, matrix,
                                world.waypoints))
        detailed_path = list(detail(path))
        barriers = tile_barriers(world.tiles_x_y[tile.x][tile.y],
                                 game.track_tile_margin, game.track_tile_size)
        passability = tile_passability(barriers, me.height, me.width)
        move.engine_power = 0.5
        if world.tick % 100 == 0:
            self.__path_plots.draw(path)
            self.__detailed_path_plots.draw(detailed_path)
            self.__tiles_barriers_plot.draw(
                linspace(0, game.track_tile_size, 20), passability)


Node = namedtuple('Node', ('x', 'y'))
Point = namedtuple('Point', ('x', 'y'))


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

    for v in chain.from_iterable(generate()):
        yield Point(matrix.x_position(v), matrix.y_position(v))


def is_direct(a, b, c):
    return a.x == b.x and b.x == c.x or a.y == b.y and b.y == c.y


def detail(path):
    def replace(index, point):
        if 0 < index < len(path) - 1:
            if not is_direct(path[index - 1], point, path[index + 1]):
                prev = array(path[index - 1])
                curr = array(point)
                next = array(path[index + 1])
                to_prev = prev - curr
                to_next = next - curr
                cos = dot(to_prev, to_next) / (norm(to_prev) * norm(to_next))
                k = 1 - (2 * cos + sqrt(2) * sqrt(1 - cos) - 2) / (2 * cos - 1)
                return [Point(*(curr + to_prev * k)),
                        Point(*(curr + to_next * k))]
        if index < len(path) - 2:
            next1 = path[index + 1]
            next2 = path[index + 2]
            if not is_direct(point, next1, next2):
                if point.x == next1.x:
                    x = next2.x + (point.x - next2.x) * 1.1
                    return [Point(x, point.y)]
                elif point.y == next1.y:
                    y = next2.y + (point.y - next2.y) * 1.1
                    return [Point(point.x, y)]
        return [point]

    return chain.from_iterable(replace(i, x) for i, x in enumerate(path))


def tile_barriers(tile_type, tile_margin, tile_size):
    low = tile_margin
    high = tile_size - low
    left = Border([low, 0], [low, tile_size], [1, 0])
    right = Border([high, 0], [high, tile_size], [-1, 0])
    top = Border([0, low], [tile_size, low], [0, 1])
    bottom = Border([0, high], [tile_size, high], [0, -1])
    left_top = Circle([0, 0], tile_margin)
    left_bottom = Circle([0, tile_size], tile_margin)
    right_top = Circle([tile_size, 0], tile_margin)
    right_bottom = Circle([tile_size, tile_size], tile_margin)
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
        self.__position = array(position)
        self.__radius = radius

    def passability(self, x, y, height, width):
        position = array((x, y))
        radius = max((height, width)) / 2
        return float(norm(self.__position - position) > self.__radius + radius)

    def __repr__(self):
        return 'Barrier(position={p}, radius={r})'.format(
            p=repr(self.__position), r=repr(self.__radius))

    def __eq__(self, other):
        return (array_equal(self.__position, other.__position) and
                self.__radius == other.__radius)


class Border(Barrier):
    def __init__(self, begin, end, normal):
        self.__begin = array(begin)
        self.__end = array(end)
        self.__normal = array(normal)

    def passability(self, x, y, height, width):
        position = array((x, y))
        to_car = position - self.__begin
        to_end = self.__end - self.__begin
        distance = sqrt(norm(to_car)**2 -
                        (dot(to_car, to_end) / norm(to_end))**2)
        if distance <= max((height, width)) / 2:
            return 0.0
        return float(dot(to_car, self.__normal) > 0)

    def __repr__(self):
        return 'Barrier(begin={b}, end={e}, normal={n})'.format(
            b=repr(self.__begin), e=repr(self.__end), n=repr(self.__normal))

    def __eq__(self, other):
        return (array_equal(self.__begin, other.__begin) and
                array_equal(self.__end, other.__end) and
                array_equal(self.__normal, other.__normal))


class PathPlots:
    __inited = False
    __points_plot = None
    __lines_plot = None

    def __init__(self):
        self.__figure, self.__axis = subplots()

    def draw(self, path):
        path_x = array([p.x for p in path])
        path_y = array([p.y for p in path])
        if self.__inited:
            self.__points_plot.set_xdata(path_x)
            self.__points_plot.set_ydata(path_y)
            self.__lines_plot.set_xdata(path_x)
            self.__lines_plot.set_ydata(path_y)
            self.__figure.canvas.draw()
        else:
            self.__axis.set_xlim([min(path_x) - 0.5, max(path_x) + 0.5])
            self.__axis.set_ylim([min(path_y) - 0.5, max(path_y) + 0.5])
            self.__points_plot = self.__axis.plot(path_x, path_y, 'o')[0]
            self.__lines_plot = self.__axis.plot(path_x, path_y, '-')[0]
            self.__inited = True


class SurfacePlot:
    def __init__(self):
        self.__figure = figure()
        self.__axis = self.__figure.add_subplot(1, 1, 1, projection='3d')

    def draw(self, limits, function):
        x, y = meshgrid(limits, limits)
        z = vectorize(function)(x, y)
        self.__axis.cla()
        self.__axis.plot_wireframe(x, y, z)
        self.__figure.canvas.draw()

from collections import namedtuple
from itertools import chain
from numpy import array, dot
from numpy.linalg import norm
from scipy.sparse.csgraph import dijkstra
from math import sqrt, pi

import matplotlib.pyplot as plot

from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
from model.TileType import TileType


Node = namedtuple('Node', ('x', 'y'))
Point = namedtuple('Point', ('x', 'y'))


class AdjacencyMatrix:
    def __init__(self, tiles):
        row_size = len(tiles)
        self.__column_size = len(tiles[0])

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
                    for x in range(self.__column_size * row_size)]

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
        return x * self.__column_size + y

    def x_position(self, index):
        return int(index / self.__column_size)

    def y_position(self, index):
        return index % self.__column_size

    @property
    def values(self):
        return self.__values


def graph_plot(matrix):
    x_max = max(matrix.x_position(x) for x in range(len(matrix.values)))
    y_max = max(matrix.y_position(x) for x in range(len(matrix.values)))
    plot.figure()
    for x, v in enumerate(matrix.values):
        s = array([matrix.x_position(x), matrix.y_position(x)])
        plot.plot([s[0]], [s[1]], 'o')
        for y, w in enumerate(v):
            if w:
                d = array([matrix.x_position(y), matrix.y_position(y)]) - s
                plot.arrow(s[0], s[1], d[0], d[1], head_width=0.2, head_length=0.2, fc='k', ec='k')
    plot.axis([-1, x_max + 1, -1, y_max + 1])
    plot.show()


def path_to_end(me: Car, world: World, game: Game):
    matrix = AdjacencyMatrix(world.tiles_x_y)
    # graph_plot(matrix)
    graph = array(matrix.values)
    _, predecessors = dijkstra(graph, return_predecessors=True)

    def generate():
        for x in range(me.next_waypoint_index - 1, len(world.waypoints) - 1):
            src = matrix.index(*world.waypoints[x])
            dst = matrix.index(*world.waypoints[x + 1])
            yield path(src, dst)

    def path(src, dst):
        return reversed(list(back_path(src, dst)))

    def back_path(src, dst):
        while src != dst and dst >= 0:
            yield dst
            dst = predecessors.item(src, dst)

    for v in chain.from_iterable(generate()):
        x = (matrix.x_position(v) + 0.5) * game.track_tile_size
        y = (matrix.y_position(v) + 0.5) * game.track_tile_size
        yield Point(x, y)


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


class MyStrategy:
    def move(self, me: Car, world: World, game: Game, move: Move):
        path = list(path_to_end(me, world, game))
        path = list(detail(path))
        target = path[0]
        move.wheel_turn = me.get_angle_to(target.x, target.y) * 20.0 / pi
        move.engine_power = 1.0
        print((me.x, me.y), tuple(target))
        # print([norm(array(path[x]) - array(path[x - 1]))
        #        for x in range(1, len(path))])
        # path_x = [x.x for x in path]
        # path_y = [x.y for x in path]
        # plot.figure()
        # plot.plot(path_x, path_y, 'o')
        # plot.plot(path_x, path_y, '-')
        # plot.axis([min(path_x) - 0.5, max(path_x) + 0.5,
        #            min(path_y) - 0.5, max(path_y) + 0.5])
        # plot.show()
        # exit(0)

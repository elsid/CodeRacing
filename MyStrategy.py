from collections import namedtuple
from itertools import chain
from numpy import array, dot
from numpy.linalg import norm
from scipy.sparse.csgraph import dijkstra
from math import sqrt, pi
# from matplotlib.pyplot import figure, ion, show

from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
from model.TileType import TileType


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
    return Point(int(x / track_tile_size - 0.5), int(y / track_tile_size - 0.5))


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

    yield Point(*waypoints[next_waypoint_index - 1])
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


class MyStrategy:
    def move(self, me: Car, world: World, game: Game, move: Move):
        matrix = AdjacencyMatrix(world.tiles_x_y)
        tile = current_tile(me.x, me.y, game.track_tile_size)
        tile_index = matrix.index(tile.x, tile.y)
        path = list(path_to_end(tile_index, me.next_waypoint_index, matrix,
                                world.waypoints))
        target = Point(*(array(path[1]) + array([0.5, 0.5]) *
                         game.track_tile_size))
        move.wheel_turn = me.get_angle_to(target.x, target.y) * 20.0 / pi
        move.engine_power = 0.5
        print('next_waypoint=', me.next_waypoint_index, me.next_waypoint_x,
              me.next_waypoint_y, *world.waypoints[me.next_waypoint_index])
        print('tile=', tile)
        print('tile_index=', tile_index)
        print('len(path)=', len(path))
        print('target=', target)
        # print((me.x, me.y), tuple(target))
        # path_x = [x.x for x in path]
        # path_y = [x.y for x in path]
        # self.path.clear()
        # self.path.set_xlim([min(path_x) - 0.5, max(path_x) + 0.5])
        # self.path.set_ylim([min(path_y) - 0.5, max(path_y) + 0.5])
        # self.path.plot(path_x, path_y, '-')
        # self.path.plot(path_x, path_y, 'o')
        # self.figure.canvas.draw()

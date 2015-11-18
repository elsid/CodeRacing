from collections import namedtuple
from numpy import array

from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
from model.TileType import TileType


Node = namedtuple('Node', ('x', 'y'))


def adjacency_matrix(tiles):
    row_size = len(tiles)
    column_size = len(tiles[0])

    def impl():
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
        elif tile == TileType.BOTTOM_HEADED_T:
            return matrix_row({left(node), right(node),
                               top(node), bottom(node)})
        else:
            return matrix_row({})

    def matrix_row(dst):
        return [1 if x in dst else 0 for x in range(column_size + row_size)]

    def left(node):
        return index(node.x - 1, node.y)

    def right(node):
        return index(node.x + 1, node.y)

    def top(node):
        return index(node.x, node.y - 1)

    def bottom(node):
        return index(node.x, node.y + 1)

    def index(x: int, y: int):
        return x * column_size + y

    return list(impl())


class MyStrategy:
    def move(self, me: Car, world: World, game: Game, move: Move):
        print(array(adjacency_matrix(world.tiles_x_y)))

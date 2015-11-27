from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
from strategy.release import ReleaseStrategy
from strategy.common import Point
from debug import Plot


class DebugStrategy:
    def __init__(self):
        self.__impl = ReleaseStrategy()
        self.__plot = Plot()

    def move(self, me: Car, world: World, game: Game, move: Move):
        self.__impl.move(me, world, game, move)
        if world.tick % 50 == 0:
            path = self.__impl.path
            tiles_path = self.__impl.tiles_path
            if path is None:
                return
            self.__plot.clear()
            self.__plot.path([Point(p.x, -p.y) for p in tiles_path], 'o')
            self.__plot.path([Point(p.x, -p.y) for p in tiles_path], '-')
            self.__plot.path([Point(p.x, -p.y) for p in path], 'o')
            self.__plot.path([Point(p.x, -p.y) for p in path], '-')
            self.__plot.draw()

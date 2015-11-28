from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
from strategy_release import ReleaseStrategy
from strategy_common import Point


class DebugStrategy:
    def __init__(self):
        from debug import Plot
        self.__impl = ReleaseStrategy()
        self.__plot = Plot()

    def move(self, me: Car, world: World, game: Game, move: Move):
        self.__impl.move(me, world, game, move, is_debug=True)
        if world.tick % 50 == 0:
            path = self.__impl.path
            position = self.__impl.position
            target = self.__impl.target_position
            tiles_path = self.__impl.tiles_path
            if path is None or target is None or position is None:
                return
            self.__plot.clear()
            self.__plot.path([Point(p.x, -p.y) for p in tiles_path], 'o')
            self.__plot.path([Point(p.x, -p.y) for p in tiles_path], '-')
            self.__plot.path([Point(p.x, -p.y) for p in path], 'o')
            self.__plot.path([Point(p.x, -p.y) for p in path], '-')
            self.__plot.path([Point(p.x, -p.y) for p in [position, target]], '-')
            self.__plot.path([Point(p.x, -p.y) for p in [position, target]], 'o')
            self.__plot.draw()

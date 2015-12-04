from os import environ
from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
from strategy_release import Context, ReleaseStrategy


class MyStrategy:
    def __init__(self):
        if 'DEBUG' in environ and environ['DEBUG'] == '1':
            from strategy_debug import DebugStrategy
            self.__impl = DebugStrategy()
        else:
            self.__impl = ReleaseStrategy()

    def move(self, me: Car, world: World, game: Game, move: Move):
        context = Context(me=me, world=world, game=game, move=move)
        if isinstance(self.__impl, ReleaseStrategy):
            try:
                self.__impl.move(context)
            except Exception:
                self.__impl = ReleaseStrategy()
            except BaseException:
                self.__impl = ReleaseStrategy()
        else:
            self.__impl.move(context)

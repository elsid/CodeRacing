from os import environ
from time import time
from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World
from strategy_release import Context, ReleaseStrategy


def profile(func):
    if 'PROFILE' in environ and environ['PROFILE'] == '1':
        def wrap(self, me: Car, world: World, game: Game, move: Move):
            start = time()
            result = func(self, me, world, game, move)
            finish = time()
            print(world.tick, finish - start)
            return result

        return wrap
    else:
        return func


class MyStrategy:
    def __init__(self):
        if 'DEBUG' in environ and environ['DEBUG'] == '1':
            from strategy_debug import DebugStrategy
            self.__impl = DebugStrategy()
        else:
            self.__impl = ReleaseStrategy()

    @profile
    def move(self, me: Car, world: World, game: Game, move: Move):
        if 'MAX_TICKS' in environ and world.tick >= int(environ['MAX_TICKS']):
            exit(0)
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

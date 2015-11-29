from strategy_release import ReleaseStrategy, Context
from strategy_common import Point
from strategy_control import Controller


class DebugStrategy:
    def __init__(self):
        from debug import Plot
        self.__impl = ReleaseStrategy(make_debug_controller)
        self.__plot = Plot()

    def move(self, context: Context):
        self.__impl.move(context)
        if context.world.tick % 50 == 0:
            path = self.__impl.path
            position = context.position
            target = self.__impl.target_position
            if path is None or target is None or position is None:
                return
            self.__plot.clear()
            self.__plot.path([Point(p.x, -p.y) for p in path], 'o')
            self.__plot.path([Point(p.x, -p.y) for p in path], '-')
            self.__plot.path([Point(p.x, -p.y) for p in [position, target]], '-')
            self.__plot.path([Point(p.x, -p.y) for p in [position]], 'o')
            self.__plot.path([Point(p.x, -p.y) for p in [target]], 'x')
            self.__plot.draw()


def make_debug_controller(context: Context):
    return Controller(
        distance_to_wheels=context.me.width / 4,
        max_engine_power_derivative=context.game.car_engine_power_change_per_tick,
        angular_speed_factor=context.game.car_angular_speed_factor,
        is_debug=True,
    )

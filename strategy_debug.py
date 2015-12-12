from os import environ
from strategy_release import ReleaseStrategy, Context
from strategy_common import Point, get_tile_center


class DebugStrategy:
    def __init__(self):
        from debug import Plot
        self.__impl = ReleaseStrategy()
        if 'PLOT' in environ and environ['PLOT'] == '1':
            self.__plot = Plot()

    def move(self, context: Context):
        self.__impl.move(context)
        if ('PLOT' in environ and environ['PLOT'] == '1' and
                context.world.tick > context.game.initial_freeze_duration_ticks and
                context.world.tick % 50 == 0):
            path = self.__impl.path
            position = context.position
            target = self.__impl.target_position
            tile_size = context.game.track_tile_size
            waypoints = [get_tile_center(Point(p[0], p[1]), tile_size)
                         for p in context.world.waypoints]
            next_waypoint = waypoints[context.me.next_waypoint_index]
            self.__plot.clear()
            if path is not None:
                self.__plot.path([Point(p.x, -p.y) for p in path], 'o')
                self.__plot.path([Point(p.x, -p.y) for p in path], '-')
            if target is not None:
                self.__plot.path([Point(p.x, -p.y) for p in [position, target]],
                                 '-')
                self.__plot.path([Point(p.x, -p.y) for p in [target]], 's')
            self.__plot.path([Point(p.x, -p.y) for p in [position]], 'o')
            self.__plot.path([Point(p.x, -p.y) for p in waypoints], 'D')
            self.__plot.path([Point(p.x, -p.y) for p in [next_waypoint]], 'D')
            self.__plot.draw()

from strategy.common import Point
from strategy.common import get_current_tile


def make_passability_function(barriers, radius, speed, tiles, tile_size):
    def impl(x, y):
        tile = get_current_tile(Point(x, y), tile_size)
        if tile not in tiles:
            return 0.0
        return min((b.passability(Point(x, y), radius, speed)
                    for b in barriers), default=1.0)
    return impl

from itertools import islice, chain
from strategy.common import Point


def shift_on_direct(path):
    if len(path) < 2:
        return (x for x in path)
    if path[0].x == path[1].x:
        last = next((i for i, p in islice(enumerate(path), 1, len(path))
                    if p.x != path[i - 1].x), len(path) - 1)
        x = path[last].x
        if x != path[0].x:
            return chain((Point(x, p.y) for p in islice(path, last)),
                         islice(path, last, len(path) - 1))
    elif path[0].y == path[1].y:
        last = next((i for i, p in islice(enumerate(path), 1, len(path))
                    if p.y != path[i - 1].y), len(path) - 1)
        y = path[last].y
        if y != path[0].y:
            return chain((Point(p.x, y) for p in islice(path, last)),
                         islice(path, last, len(path) - 1))
    return (x for x in path)

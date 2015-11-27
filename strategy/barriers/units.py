from model.RectangularUnit import RectangularUnit
from model.CircularUnit import CircularUnit
from strategy.common import Point
from strategy.barriers.unit import Unit


def make_units_barriers(units):
    return [unit_barriers(x) for x in units]


def unit_barriers(unit):
    if isinstance(unit, RectangularUnit):
        radius = min((unit.height, unit.width)) / 2
    elif isinstance(unit, CircularUnit):
        radius = unit.radius
    else:
        radius = 1.0
    return Unit(position=Point(unit.x, unit.y), radius=radius,
                speed=Point(unit.speed_x, unit.speed_y))

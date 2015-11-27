from strategy.barriers.circle import Circle


class Unit:
    def __init__(self, position, radius, speed):
        self.__circle = Circle(position, radius)
        self.__position = position
        self.__radius = radius
        self.__speed = speed

    def passability(self, position, radius, speed):
        immovable = self.__circle.passability(position, radius, speed)
        if immovable == 1.0:
            return 1.0
        else:
            distance = (self.__position - position).norm()
            return (distance / (self.__radius + radius)) ** 2

    def __repr__(self):
        return 'Unit(position={p}, radius={r}, speed={s})'.format(
            p=repr(self.__position), r=repr(self.__radius),
            s=repr(self.__speed))

    def __eq__(self, other):
        return (self.__position == other.__position and
                self.__radius == other.__radius and
                self.__speed == other.__speed)

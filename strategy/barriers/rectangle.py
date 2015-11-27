from strategy.common import Point, Line


class Rectangle:
    INSIDE = 0
    LEFT = 1
    RIGHT = 2
    TOP = 4
    BOTTOM = 8

    def __init__(self, left_top, right_bottom):
        self.left_top = left_top
        self.right_bottom = right_bottom

    def __repr__(self):
        return 'Rectangle(left_top={lt}, right_bottom={rb})'.format(
            lt=repr(self.left_top), rb=repr(self.right_bottom))

    def __eq__(self, other):
        return (self.left_top == other.left_top and
                self.right_bottom == other.right_bottom)

    def passability(self, position, radius, _=None):
        position_code = self.point_code(position)
        if position_code == Rectangle.INSIDE:
            return 0.0
        width = self.right_bottom.x - self.left_top.x
        height = self.right_bottom.y - self.left_top.y
        center = self.left_top + Point(width / 2, height / 2)
        direction = center - position
        border = position + direction / direction.norm() * radius
        border_code = self.point_code(border)
        return float(position_code & border_code)

    def point_code(self, point):
        result = Rectangle.INSIDE
        if point.x < self.left_top.x:
            result |= Rectangle.LEFT
        elif point.x > self.right_bottom.x:
            result |= Rectangle.RIGHT
        if point.y < self.left_top.y:
            result |= Rectangle.TOP
        elif point.y > self.right_bottom.y:
            result |= Rectangle.BOTTOM
        return result

    def left(self):
        return Line(begin=self.left_top,
                    end=self.left_top + Point(0, self.height()))

    def right(self):
        return Line(begin=self.right_bottom,
                    end=self.right_bottom - Point(0, self.height()))

    def top(self):
        return Line(begin=self.left_top + Point(self.width(), 0),
                    end=self.left_top)

    def bottom(self):
        return Line(begin=self.right_bottom - Point(self.width(), 0),
                    end=self.right_bottom)

    def width(self):
        return self.right_bottom.x - self.left_top.x

    def height(self):
        return self.right_bottom.y - self.left_top.y

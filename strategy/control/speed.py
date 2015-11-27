from strategy.common import Point


def get_speed(position: Point, following: Point, after_following: Point,
              my_direction: Point):
    direction = (after_following - following).normalized()
    to_following = following - position
    to_after_following = after_following - following
    return (direction * get_speed_gain(to_following.cos(to_after_following) *
                                       my_direction.cos(to_following)) +
            to_following / 400)


def get_speed_gain(x):
    return 1 - 3 / (x - 1)

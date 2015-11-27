from strategy.path.reduce_direct import reduce_base_on_three


def reduce_diagonal_direct(path):
    return reduce_base_on_three(path, is_diagonal_direct)


def is_diagonal_direct(previous, current, following):
    to_previous = previous - current
    to_following = following - current
    return to_following.x == -to_previous.x and to_following.y == -to_previous.y

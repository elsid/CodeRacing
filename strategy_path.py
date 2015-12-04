from enum import Enum
from collections import namedtuple
from itertools import islice, chain, groupby
from numpy import array
from sys import maxsize
from scipy.sparse.csgraph import dijkstra
from collections import defaultdict, deque
from heapq import heappop, heappush
from model.TileType import TileType
from strategy_common import Point, get_current_tile


def adjust_path(path, shift):
    if len(path) < 2:
        return (x for x in path)

    def generate():
        typed_path = list(make_typed_path())
        yield adjust_path_point(None, typed_path[0], typed_path[1], shift)
        for i, p in islice(enumerate(typed_path), 1, len(typed_path) - 1):
            yield adjust_path_point(typed_path[i - 1], p, typed_path[i + 1],
                                    shift)
        yield adjust_path_point(typed_path[-2], typed_path[-1], None, shift)

    def make_typed_path():
        yield TypedPoint(path[0],
                         PointType(SideType.UNKNOWN,
                                   output_type(path[0], path[1])))
        for i, p in islice(enumerate(path), 1, len(path) - 1):
            yield TypedPoint(p, point_type(path[i - 1], p, path[i + 1]))
        yield TypedPoint(path[-1],
                         PointType(input_type(path[-2], path[-1]),
                                   SideType.UNKNOWN))

    return generate()


TypedPoint = namedtuple('TypedPoint', ('position', 'type'))


def adjust_path_point(previous, current: TypedPoint, following, shift):
    return current.position + path_point_shift(previous, current, following,
                                               shift)


def path_point_shift(previous: TypedPoint, current: TypedPoint,
                     following: TypedPoint, shift):
    if current.type in {PointType.LEFT_TOP, PointType.TOP_LEFT}:
        if following and current.type.input == following.type.output:
            return Point(+ shift, + shift) / 2
        else:
            return Point(- shift, - shift)
    elif current.type in {PointType.LEFT_BOTTOM, PointType.BOTTOM_LEFT}:
        if following and current.type.input == following.type.output:
            return Point(+ shift, - shift) / 2
        else:
            return Point(- shift, + shift)
    elif current.type in {PointType.RIGHT_TOP, PointType.TOP_RIGHT}:
        if following and current.type.input == following.type.output:
            return Point(- shift, + shift) / 2
        else:
            return Point(+ shift, - shift)
    elif current.type in {PointType.RIGHT_BOTTOM, PointType.BOTTOM_RIGHT}:
        if following and current.type.input == following.type.output:
            return Point(- shift, - shift) / 2
        else:
            return Point(+ shift, + shift)
    elif current.type in {PointType.LEFT_RIGHT, PointType.RIGHT_LEFT}:
        if following and following.type.output == SideType.TOP:
            return Point(0, + shift)
        elif following and following.type.output == SideType.BOTTOM:
            return Point(0, - shift)
    elif current.type in {PointType.TOP_BOTTOM, PointType.BOTTOM_TOP}:
        if following and following.type.output == SideType.LEFT:
            return Point(+ shift, 0)
        elif following and following.type.output == SideType.RIGHT:
            return Point(- shift, 0)
    return Point(0, 0)


def point_type(previous, current, following):
    return PointType(input_type(previous, current),
                     output_type(current, following))


def input_type(previous, current):
    if previous.y == current.y:
        if previous.x < current.x:
            return SideType.LEFT
        else:
            return SideType.RIGHT
    if previous.x == current.x:
        if previous.y < current.y:
            return SideType.TOP
        else:
            return SideType.BOTTOM
    return SideType.UNKNOWN


def output_type(current, following):
    if current.y == following.y:
        if current.x < following.x:
            return SideType.RIGHT
        else:
            return SideType.LEFT
    if current.x == following.x:
        if current.y < following.y:
            return SideType.BOTTOM
        else:
            return SideType.TOP
    return SideType.UNKNOWN


class SideType(Enum):
    UNKNOWN = 0
    LEFT = 1
    RIGHT = 2
    TOP = 3
    BOTTOM = 4


PointTypeImpl = namedtuple('PointType', ('input', 'output'))


class PointType(PointTypeImpl):
    LEFT_RIGHT = PointTypeImpl(SideType.LEFT, SideType.RIGHT)
    LEFT_TOP = PointTypeImpl(SideType.LEFT, SideType.TOP)
    LEFT_BOTTOM = PointTypeImpl(SideType.LEFT, SideType.BOTTOM)
    RIGHT_LEFT = PointTypeImpl(SideType.RIGHT, SideType.LEFT)
    RIGHT_TOP = PointTypeImpl(SideType.RIGHT, SideType.TOP)
    RIGHT_BOTTOM = PointTypeImpl(SideType.RIGHT, SideType.BOTTOM)
    TOP_LEFT = PointTypeImpl(SideType.TOP, SideType.LEFT)
    TOP_RIGHT = PointTypeImpl(SideType.TOP, SideType.RIGHT)
    TOP_BOTTOM = PointTypeImpl(SideType.TOP, SideType.BOTTOM)
    BOTTOM_LEFT = PointTypeImpl(SideType.BOTTOM, SideType.LEFT)
    BOTTOM_RIGHT = PointTypeImpl(SideType.BOTTOM, SideType.RIGHT)
    BOTTOM_TOP = PointTypeImpl(SideType.BOTTOM, SideType.TOP)


def reduce_diagonal_direct(path):
    return reduce_base_on_three(path, is_diagonal_direct)


def is_diagonal_direct(previous, current, following):
    to_previous = previous - current
    to_following = following - current
    return to_following.x == -to_previous.x and to_following.y == -to_previous.y


def reduce_direct_first_after_me(path):
    if len(path) < 2:
        return (x for x in path)
    following = path[0]
    after_following = path[1]
    if following.x == after_following.x or following.y == after_following.y:
        return islice(path, 1, len(path))
    return (x for x in path)


def reduce_direct(path):
    return reduce_base_on_three(path, is_direct)


def is_direct(previous, current, following):
    return (current.x == previous.x and current.x == following.x or
            current.y == previous.y and current.y == following.y)


def reduce_base_on_three(path, need_reduce):
    if not path:
        return []
    yield path[0]
    if len(path) == 1:
        return
    for i, current in islice(enumerate(path), 1, len(path) - 1):
        if not need_reduce(path[i - 1], current, path[i + 1]):
            yield current
    yield path[-1]


def shift_on_direct(path):
    if len(path) < 2:
        return (x for x in path)
    last = 1
    yield path[0]
    while last < len(path) - 1:
        if (path[last - 1].x == path[last].x and
                path[last].x == path[last + 1].x):
            shift, points = shift_on_direct_x(path[last:])
            last += shift
            for p in points:
                yield p
        elif (path[last - 1].y == path[last].y and
                path[last].y == path[last + 1].y):
            shift, points = shift_on_direct_y(path[last:])
            last += shift
            for p in points:
                yield p
        yield path[last]
        last += 1
    if last < len(path):
        yield path[last]


def shift_on_direct_x(path):
    last = next((i for i, p in islice(enumerate(path), 1, len(path))
                if p.x != path[i - 1].x), len(path) - 1)
    x = path[last].x
    if x != path[0].x:
        return last, (Point(x, p.y) for p in islice(path, last))
    return last, (p for p in path)


def shift_on_direct_y(path):
    last = next((i for i, p in islice(enumerate(path), 1, len(path))
                 if p.y != path[i - 1].y), len(path) - 1)
    y = path[last].y
    if y != path[0].y:
        return last, (Point(p.x, y) for p in islice(path, last))
    return last, (p for p in path)


def shift_to_borders(path):
    if not path:
        return []
    for i, current in islice(enumerate(path), len(path) - 1):
        following = path[i + 1]
        direction = following - current
        yield current + direction * 0.5
    yield path[-1]


# def make_tiles_path(start_tile, waypoints,  tiles,
#                     direction):
#     matrix = AdjacencyMatrix(tiles, start_tile, direction)
#     tile_index = matrix.index(start_tile.x, start_tile.y)
#     return make_path(tile_index, matrix, waypoints)


def make_path(start_index, matrix, waypoints):
    graph = array(matrix.values)
    _, predecessors = dijkstra(graph, return_predecessors=True)

    def generate():
        yield path(start_index, matrix.index(*waypoints[0]))
        for i, p in islice(enumerate(waypoints), len(waypoints) - 1):
            src = matrix.index(*p)
            dst = matrix.index(*waypoints[i + 1])
            yield path(src, dst)

    def path(src, dst):
        return reversed(list(back_path(src, dst)))

    def back_path(src, dst):
        while src != dst and dst >= 0:
            yield dst
            dst = predecessors.item(src, dst)

    yield matrix.point(start_index)
    for v in chain.from_iterable(generate()):
        yield matrix.point(v)


class AdjacencyMatrix:
    def __init__(self, tiles, start_tile, direction):
        column_size = len(tiles)
        self.__row_size = len(tiles[0])

        def generate():
            for x, column in enumerate(tiles):
                for y, tile in enumerate(column):
                    yield list(adjacency_matrix_row(Point(x, y), tile))

        def adjacency_matrix_row(node, tile):
            def matrix_row(dst):
                for x in range(self.__row_size * column_size):
                    if x in dst:
                        distance = (0.5 if node == start_tile
                                    else node.distance(start_tile))
                        yield 3 - ((self.point(x) - node).cos(direction) /
                                   distance)
                    else:
                        yield 0

            if tile == TileType.VERTICAL:
                return matrix_row({top(node), bottom(node)})
            elif tile == TileType.HORIZONTAL:
                return matrix_row({left(node), right(node)})
            elif tile == TileType.LEFT_TOP_CORNER:
                return matrix_row({right(node), bottom(node)})
            elif tile == TileType.RIGHT_TOP_CORNER:
                return matrix_row({left(node), bottom(node)})
            elif tile == TileType.LEFT_BOTTOM_CORNER:
                return matrix_row({right(node), top(node)})
            elif tile == TileType.RIGHT_BOTTOM_CORNER:
                return matrix_row({left(node), top(node)})
            elif tile == TileType.LEFT_HEADED_T:
                return matrix_row({left(node), top(node), bottom(node)})
            elif tile == TileType.RIGHT_HEADED_T:
                return matrix_row({right(node), top(node), bottom(node)})
            elif tile == TileType.TOP_HEADED_T:
                return matrix_row({top(node), left(node), right(node)})
            elif tile == TileType.BOTTOM_HEADED_T:
                return matrix_row({bottom(node), left(node), right(node)})
            elif tile == TileType.CROSSROADS:
                return matrix_row({left(node), right(node),
                                   top(node), bottom(node)})
            else:
                return matrix_row({})

        def left(node):
            return self.index(node.x - 1, node.y)

        def right(node):
            return self.index(node.x + 1, node.y)

        def top(node):
            return self.index(node.x, node.y - 1)

        def bottom(node):
            return self.index(node.x, node.y + 1)

        self.__values = list(generate())

    def index(self, x, y):
        return x * self.__row_size + y

    def x_position(self, index):
        return int(index / self.__row_size)

    def y_position(self, index):
        return index % self.__row_size

    def point(self, index):
        return Point(self.x_position(index), self.y_position(index))

    @property
    def values(self):
        return self.__values


def make_tiles_path(start_tile, waypoints, tiles, direction):
    graph = make_graph(tiles)
    graph = split_arcs(graph)
    graph = add_diagonal_arcs(graph)
    row_size = len(tiles[0])
    start = get_point_index(start_tile, row_size)
    waypoints = [get_index(x[0], x[1], row_size) for x in waypoints]
    if start != waypoints[0] and start in graph:
        waypoints = [start] + waypoints
    path = multi_path(graph, waypoints, direction)
    return remove_split(list(graph[x].position + Point(0.5, 0.5) for x in path))


def multi_path(graph, waypoints, direction):
    if len(waypoints) < 2:
        return []
    path = [waypoints[0]]
    for i, w in islice(enumerate(waypoints), 0, len(waypoints) - 1):
        if w in graph and waypoints[i + 1] in graph:
            path += list(shortest_path_with_direction(
                graph, w, waypoints[i + 1], direction))
        if len(path) > 2:
            direction = graph[path[-1]].position - graph[path[-2]].position
    return path


Node = namedtuple('Node', ('position', 'arcs'))
Arc = namedtuple('Arc', ('dst', 'weight'))


def make_graph(tiles):
    row_size = len(tiles[0])

    def left(pos):
        return get_index(pos.x - 1, pos.y, row_size)

    def right(pos):
        return get_index(pos.x + 1, pos.y, row_size)

    def top(pos):
        return get_index(pos.x, pos.y - 1, row_size)

    def bottom(pos):
        return get_index(pos.x, pos.y + 1, row_size)

    def tile_arcs(pos, tile_type):
        if tile_type == TileType.VERTICAL:
            return top(pos), bottom(pos)
        elif tile_type == TileType.HORIZONTAL:
            return left(pos), right(pos)
        elif tile_type == TileType.LEFT_TOP_CORNER:
            return right(pos), bottom(pos)
        elif tile_type == TileType.RIGHT_TOP_CORNER:
            return left(pos), bottom(pos)
        elif tile_type == TileType.LEFT_BOTTOM_CORNER:
            return right(pos), top(pos)
        elif tile_type == TileType.RIGHT_BOTTOM_CORNER:
            return left(pos), top(pos)
        elif tile_type == TileType.LEFT_HEADED_T:
            return left(pos), top(pos), bottom(pos)
        elif tile_type == TileType.RIGHT_HEADED_T:
            return right(pos), top(pos), bottom(pos)
        elif tile_type == TileType.TOP_HEADED_T:
            return top(pos), left(pos), right(pos)
        elif tile_type == TileType.BOTTOM_HEADED_T:
            return bottom(pos), left(pos), right(pos)
        elif tile_type == TileType.CROSSROADS:
            return left(pos), right(pos), top(pos), bottom(pos)
        else:
            return tuple()

    result = {}
    for x, column in enumerate(tiles):
        for y, tile in enumerate(column):
            position = Point(x, y)
            node = Node(position, [])
            result[get_index(x, y, row_size)] = node
            for index in tile_arcs(node.position, tile):
                neighbor = result.get(index)
                if neighbor is None:
                    neighbor_position = get_point(index, row_size)
                    neighbor = Node(neighbor_position, [])
                    result[index] = neighbor
                else:
                    neighbor_position = neighbor.position
                weight = position.distance(neighbor_position)
                node.arcs.append(Arc(index, weight))
    return result


def split_arcs(graph):
    node_ids = iter(range(len(graph), maxsize))
    middles = {}
    result = {}
    for index, node in graph.items():
        for arc in node.arcs:
            middle_id = middles.get((index, arc.dst))
            dst = graph[arc.dst]
            if index not in result:
                result[index] = Node(node.position, [])
            if arc.dst not in result:
                result[arc.dst] = Node(dst.position, [])
            if middle_id is None:
                middle_id = next(node_ids)
                middles[(arc.dst, index)] = middle_id
                result[index].arcs.append(Arc(middle_id, arc.weight / 2))
                result[arc.dst].arcs.append(Arc(middle_id, arc.weight / 2))
                result[middle_id] = Node((node.position + dst.position) / 2,
                                         [Arc(index, arc.weight / 2),
                                          Arc(arc.dst, arc.weight / 2)])
    return result


def add_diagonal_arcs(graph):
    def new_arcs():
        result = defaultdict(list)
        for index, node in graph.items():
            arcs = zip(islice(node.arcs, len(node.arcs) - 1),
                       islice(node.arcs, 1, len(node.arcs)))
            for first_arc, second_arc in arcs:
                first_node = graph[first_arc.dst]
                second_node = graph[second_arc.dst]
                first_direction = first_node.position - node.position
                second_direction = second_node.position - node.position
                distance = first_node.position.distance(second_node.position)
                if ((first_arc.dst, second_arc.dst) not in result and
                        (second_arc.dst, first_arc.dst) not in result and
                        first_direction.cos(second_direction) >= 0):
                    result[first_arc.dst].append(Arc(second_arc.dst, distance))
                    result[second_arc.dst].append(Arc(first_arc.dst, distance))
        return result

    def generate():
        arcs = new_arcs()
        for index, node in graph.items():
            yield index, Node(node.position, node.arcs + arcs[index])

    return dict(generate())


def get_point_index(point, row_size):
    return get_index(point.x, point.y, row_size)


def get_index(x, y, row_size):
    return x * row_size + y


def get_point(index, row_size):
    return Point(int(index / row_size), index % row_size)


def shortest_path_with_direction(graph, src, dst, initial_direction):
    initial_direction = initial_direction.normalized()
    queue = [(0, src, initial_direction)]
    distances = {src: 0}
    previous_nodes = {}
    visited = defaultdict(list)
    while queue:
        distance, node_index, direction = heappop(queue)
        visited[node_index].append(direction)
        node = graph[node_index]
        for neighbor_index, weight in node.arcs:
            direction_from = graph[neighbor_index].position - node.position
            if (direction_from in visited[neighbor_index] or
                    direction_from.norm() == 0):
                continue
            new_direction = direction_from
            current_distance = distances.get(neighbor_index, float('inf'))
            cos_value = direction.cos(new_direction)
            if direction.cos(new_direction) < -1e-3:
                continue
            elif direction.cos(new_direction) < 1e-3:
                new_distance = distance + (2 * (1 - cos_value) + 1) * weight
            else:
                new_distance = distance + (1 * (1 - cos_value) + 1) * weight
            if new_distance < current_distance:
                distances[neighbor_index] = new_distance
                previous_nodes[neighbor_index] = node_index
                heappush(queue, (new_distance, neighbor_index, new_direction))
    result = deque()
    node_index = dst
    while node_index is not None:
        result.appendleft(node_index)
        previous = previous_nodes.get(node_index)
        node_index = previous
    if result[0] != src:
        return []
    return islice(result, 1, len(result))


def remove_split(path):
    def generate():
        for i, p in islice(enumerate(path), len(path) - 1):
            middle = (p + path[i + 1]) / 2
            yield get_current_tile(middle, 1)

    return (x[0] for x in groupby(generate()))

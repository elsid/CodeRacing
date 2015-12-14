from argparse import ArgumentParser, FileType
from collections import namedtuple
from json import loads
from matplotlib.pyplot import hist, show
from sys import stdin
from statistics import mean, median
from itertools import groupby
from operator import attrgetter


Value = namedtuple('Value', ('tick', 'id', 'time'))


def main():
    args = parse_args()
    values = read(args.file)
    ordered = sorted(values, key=attrgetter('id'))
    grouped = groupby(ordered, key=attrgetter('id'))
    for car_id, values in grouped:
        values = sorted(values, key=attrgetter('tick'))
        print(car_id, 'sum:', sum(v.time for v in values))
        print(car_id, 'min:', min(v.time for v in values))
        print(car_id, 'max:', max(v.time for v in values))
        print(car_id, 'mean:', mean(v.time for v in values))
        print(car_id, 'median:', median(v.time for v in values))
        hist([v.time for v in values], 100)
    show()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('file', type=FileType('r'), default=stdin)
    return parser.parse_args()


def read(stream):
    for line in stream:
        data = loads(line)
        yield Value(**data)


if __name__ == '__main__':
    main()

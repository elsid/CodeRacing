from debug.plot import Plot


def log(**kwargs):
    from json import dumps
    print(dumps(kwargs))

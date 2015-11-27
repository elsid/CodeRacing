from numpy import meshgrid, vectorize


class Plot:
    def __init__(self, title=None):
        from matplotlib.pyplot import figure, ion, show
        self.__figure = figure()
        self.__axis = self.__figure.add_subplot(1, 1, 1)
        self.__title = title
        ion()
        show()

    def clear(self):
        self.__axis.cla()

    def surface(self, x, y, function):
        from mpl_toolkits.mplot3d import Axes3D
        x, y = meshgrid(x, y)
        z = vectorize(function)(x, y)
        self.__axis.imshow(z, alpha=0.5,
                           extent=[x.min(), x.max(), y.min(), y.max()])

    def path(self, points, *args, **kwargs):
        x = [p.x for p in points]
        y = [p.y for p in points]
        self.__axis.plot(x, y, *args, **kwargs)

    def curve(self, x, function, *args, **kwargs):
        y = vectorize(function)(x)
        self.__axis.plot(x, y, *args, **kwargs)

    def lines(self, x, y, *args, **kwargs):
        self.__axis.plot(x, y, *args, **kwargs)

    def draw(self):
        if self.__title is not None:
            self.__axis.set_title(self.__title)
        self.__figure.canvas.draw()

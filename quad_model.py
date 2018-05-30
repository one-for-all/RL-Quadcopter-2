import numpy as np


class Model:
    def __init__(self, ax, length=1):
        colors = ['r', 'g', 'b']
        self.ax = ax
        self.lines = sum([ax.plot([], [], [], c=c) for c in colors], [])
        self.pos = np.array([0, 0, 0])
        self.start_points = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.end_points = np.array([[length, 0, 0], [0, length, 0], [0, 0, length]])
        for line, start, end in zip(self.lines, self.start_points,
                                    self.end_points):
            line.set_data((start[0], end[0]), (start[1], end[1]))
            line.set_3d_properties((start[2], end[2]))

        def generate_quaternion():
            q1 = Quaternion.random()
            q2 = Quaternion.random()
            while True:
                for q in Quaternion.intermediates(q1, q2, 20,
                                                  include_endpoints=True):
                    yield q
                q1 = q2
                q2 = Quaternion.random()
        self.quaternion_generator = generate_quaternion()

    def random_rotate(self):
        q = next(self.quaternion_generator)
        self.rotate(q)

    def rotate(self, q):
        for line, start, end in zip(self.lines, self.start_points,
                                    self.end_points):
            start = q.rotate(start)
            end = q.rotate(end)
            start_pos = start + self.pos
            end_pos = end + self.pos
            line.set_data([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]])
            line.set_3d_properties([start_pos[2], end_pos[2]])

    def set_pos(self, pos):
        self.pos = pos
        for line, start, end in zip(self.lines, self.start_points,
                                    self.end_points):
            start_pos = start + self.pos
            end_pos = end + self.pos
            line.set_data([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]])
            line.set_3d_properties([start_pos[2], end_pos[2]])

    def redraw(self):
        self.ax.draw_artist(self.ax.patch)
        for line in self.lines:
            self.ax.draw_artist(line)
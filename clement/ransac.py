import numpy as np


class Ransac():
    def __init__(self, x, y, n, rad):
        self.x = x
        self.y = y
        self.rad = rad
        self.n = n
        self.err_max = 99999
        self.best_fit = None

    def get_subsample(self):
        sample = []
        indices = []
        while len(sample) < 3:
            idx = np.random.randint(len(self.x))
            if idx not in indices:
                point = np.array((self.x[idx], self.y[idx]))
                sample.append(point)
                indices.append(idx)
        return sample

    def calc_center(self, sample):
        A = np.array([sample[1] - sample[0], sample[2] - sample[1]])
        B = np.array([(sample[1] ** 2 - sample[0] ** 2).sum(), (sample[2] ** 2 - sample[1] ** 2).sum()])
        c_x, c_y = np.linalg.inv(A) @ B / 2
        return c_x, c_y

    def check_model(self, model):
        err = 0
        c_x, c_y = model
        for i in range(len(self.x)):
            d = np.sqrt((self.x[i] - c_x) ** 2 + (self.y[i] - c_y) ** 2)
            err += np.abs(d - self.rad)
        return err

    def run(self):
        for i in range(self.n):
            center = self.calc_center(self.get_subsample())
            err = self.check_model(center)
            if err < self.err_max:
                self.best_fit = center
                self.err_max = err

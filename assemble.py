import sys
import glob
import os
import numpy as np
import scipy.signal as sc
import mrcfile as mrc
import pyqtgraph as pg
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import pyqtgraph as pg
matplotlib.use('QT5Agg')

class Assembler():
    def __init__(self, step=12):
        self.step = step

    def parse(self, fname):
        with mrc.open(fname, 'r', permissive=True) as f:
            self.data = f.data[:,::self.step,::self.step]
            self._h = f.header
            self._eh = np.frombuffer(f.extended_header, dtype='i2')

    def assemble(self):
        dimensions = self.data.shape

        pos_x = self._eh[1:10*dimensions[0]:10] // self.step
        pos_y = self._eh[2:10*dimensions[0]:10] // self.step
        pos_z = self._eh[3:10*dimensions[0]:10]

        cy, cx = np.indices(dimensions[1:3])

        self.merged = np.zeros((np.max(pos_x)+dimensions[2],np.max(pos_y)+dimensions[1]), dtype='f4')
        self.mcounts = np.zeros_like(self.merged)
        for i in range(dimensions[0]):
        #for i in range(5):
            print('Merge for image {}'.format(i))
            np.add.at(self.mcounts, (cx+pos_x[i], cy+pos_y[i]), 1)
            np.add.at(self.merged, (cx+pos_x[i], cy+pos_y[i]), self.data[i])

        self.merged[self.mcounts>0] /= self.mcounts[self.mcounts>0]

        return self.merged

    def save_merge(self, fname):
        with mrc.new(fname, overwrite=True) as f:
            f.set_data(self.merged)
            f.update_header_stats()

if __name__=='__main__':
    path = '../gs.mrc'
    assembler = Assembler()
    assembler.parse(path)
    assembler.assemble()
    pg.show(merged.T)

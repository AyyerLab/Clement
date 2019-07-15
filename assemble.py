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

class assemble(self,my_path,step):
    
    def __init__(self):

        self.step = step
        self.path = my_path
        

    def assemble(self):
    
        f = mrc.open(self.path,'r',permissive=True)
        h = f.header
        data = f.data[:,::10,::10]
        dimensions=data.shape

        eh = np.frombuffer(f.extended_header,dtype='i2')

        pos_x = eh[1:self.step*dimensions[0]:self.step]//10
        pos_y = eh[2:self.step*dimensions[0]:self.step]//10
        pos_z = eh[3:self.step*dimensions[0]:self.step]

        cy, cx = np.indices(dimensions[1:3])

        merged = np.zeros((np.max(pos_x)+dimensions[2],np.max(pos_y)+dimensions[1]))
        counts = np.zeros_like(merged)
        for i in range(dimensions[0]):
        #for i in range(5):
            print('Merge for image {}'.format(i))
            np.add.at(counts,(cx+pos_x[i],cy+pos_y[i]),1)
            np.add.at(merged,(cx+pos_x[i],cy+pos_y[i]),data[i])

        merged[counts>0] /= counts[counts>0]

        return merged

if __name__=='__main__':
    path = '../gs.mrc'
    merged = assemble(path)
    pg.show(merged.T)















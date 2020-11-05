import read_lif
from peak_finding import Peak_finding
import numpy as np
from scipy import ndimage as ndi
from matplotlib import pyplot as plot
import pyqtgraph as pg
from PyQt5 import QtCore
from scipy.optimize import curve_fit
import numpy.ma as ma

fname = '../data/3D/201029_CLEMENT_EXP/201027_CLEMENT_FM/201027_CLEMENT_Grid2.lif'
base_reader = read_lif.Reader(fname)
reader = base_reader.getSeries()
data = reader.getFrame(channel=0, dtype='u2').astype('f8')
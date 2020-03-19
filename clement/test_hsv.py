import sys
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
import numpy as np
import pyqtgraph as pg
import scipy
from scipy import signal as sc
from scipy import ndimage as ndi
from skimage.color import hsv2rgb
from skimage import transform as tf
from skimage import measure, morphology
import read_lif
import functools
plt.ion()


def wshed_peaks(img, threshold=0):
    labels = morphology.label(img >= threshold, connectivity=1)
    morphology.remove_small_objects(labels, 50, connectivity=1, in_place=True)
    print('Number of peaks found: ', labels.max())
    wshed = morphology.watershed(-img * (labels > 0), labels)
    coor = np.round(np.array([r.weighted_centroid for r in measure.regionprops((labels > 0) * wshed, img)]))
    return coor


fname = '/home/tamme/phd/Clement/data/3D/grid1_05.lif'

base_reader = read_lif.Reader(fname)
reader = base_reader.getSeries()[0]

num_slices = reader.getFrameShape()[0]
num_channels = len(reader.getChannels())

data = np.array(reader.getFrame(channel=3, dtype='u2').astype('f4'))
offset = np.array(reader.getFrame(channel=0, dtype='u2').astype('f4'))

max_proj = data.max(2) #- offset.max(2)
max_proj /= max_proj.mean((0,1))
max_proj[max_proj<4] = 0
#pg.show(max_proj)

argmax_map = np.argmax(data, axis=2).astype('f4')
argmax_map /= argmax_map.max()* 2
hsv_map = np.array([argmax_map, np.ones_like(argmax_map), max_proj])
hsv_map = hsv_map.transpose(1,2,0)
pg.show(hsv2rgb(hsv_map))


avg_max = np.sort(max_proj.ravel())[-100:].mean()
coor = wshed_peaks(max_proj, threshold=0.35*avg_max)

z_profile = data[coor[:, 0].astype(int), coor[:, 1].astype(int)]
z_max = np.argmax(z_profile, axis=1)

coor_3d = np.concatenate((coor, np.expand_dims(z_max, axis=1)), axis=1)
fig = plt.figure()
plt.scatter(coor_3d[:,0], coor_3d[:,1], c=coor_3d[:,2])
plt.colorbar()
plt.show()



A = np.array([coor_3d[:, 0], coor_3d[:, 1], np.ones_like(coor_3d[:, 0])]).T
beta = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), np.expand_dims(coor_3d[:, 2], axis=1))
print('beta: ', beta)

no_trend = coor_3d[:, 2] - (beta[0] * coor_3d[:, 0] + beta[1] * coor_3d[:, 1] + beta[2])

plt.figure()
plt.scatter(coor_3d[:,0], coor_3d[:,1], c=no_trend)
plt.colorbar()
plt.show()

point = np.array([0.0, 0.0, beta[2]])
normal = np.array(np.cross([1, 0, beta[0][0]], [0, 1, beta[1][0]]))
d = -point.dot(normal)
xx, yy = np.meshgrid([0, 2048], [0, 2048])
z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

x,y = np.indices((2048,2048))
z_all = -(normal[0]*x + normal[1]*y + d)/normal[2] #z_all gives exactly the same plane as z

z_max_all = np.argmax(data, axis=2)

no_trend_all = z_max_all-z_all
no_trend_beads = no_trend_all[coor_3d[:,0].astype(int), coor_3d[:,1].astype(int)]

plt.figure()
plt.scatter(coor_3d[:,0], coor_3d[:,1],c=no_trend_beads)
plt.colorbar()
plt.show()


no_trend_all /= no_trend_all.max() * 2
hsv_new = np.array([no_trend_all, np.ones_like(argmax_map), max_proj])
hsv_new = hsv_new.transpose(1,2,0)
pg.show(hsv2rgb(hsv_new))




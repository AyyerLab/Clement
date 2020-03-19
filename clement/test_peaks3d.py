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
from skimage import transform as tf
from skimage import measure, morphology
import read_lif
import functools
import time

plt.ion()


def peak_finding(img, threshold=0, roi=False, label_min_size=1, label_max_size=1000):
    img[img < threshold] = 0
    labels, num_obj = ndi.label(img)
    label_size = np.bincount(labels.ravel())
    mask = np.where((label_size >= label_min_size) & (label_size < label_max_size), True, False)
    label_mask = mask[labels.ravel()].reshape(labels.shape)
    labels_red, num_red = ndi.label(label_mask * labels)
    print('num_red: ', num_red)

    if roi:
        coor = np.array(ndi.center_of_mass(img, labels, labels.max()))
    else:
        coor = np.array(ndi.center_of_mass(img, labels_red, range(num_red)))

    return coor


def wshed_peaks(img, threshold=0):
    labels = morphology.label(img >= threshold, connectivity=1)
    morphology.remove_small_objects(labels, 50, connectivity=1, in_place=True)
    print('Number of peaks found: ', labels.max())
    wshed = morphology.watershed(-img * (labels > 0), labels, compactness=0.5, watershed_line=True)
    coor = np.round(np.array([r.weighted_centroid for r in measure.regionprops((labels > 0) * wshed, img)]))
    return coor


def plane(x, y, params):
    return params[0] * x + params[1] * y + params[2]


def error(params, points):
    result = 0
    for (x, y, z) in points:
        plane_z = plane(x, y, params)
        diff = abs(plane_z - z)
        result += diff ** 2
    return result


fname = '/home/tamme/phd/Clement/data/3D/grid1_05.lif'

base_reader = read_lif.Reader(fname)
reader = base_reader.getSeries()[0]

num_slices = reader.getFrameShape()[0]
num_channels = len(reader.getChannels())

data = np.array(reader.getFrame(channel=3, dtype='u2').astype('f4'))
max_proj = data.max(2)
max_proj /= max_proj.mean((0, 1))

pg.show(max_proj)

start = time.time()
t1 = np.mean(np.sort(max_proj.flatten())[-100:])

coor = wshed_peaks(max_proj, 0.3 * t1)

z_profile = data[coor[:, 0].astype(int), coor[:, 1].astype(int)]

z_max = np.argmax(z_profile, axis=1)

#plt.figure()
#plt.hist(z_max, bins=len(z_max))
#plt.title('Z profile histogram')
#plt.xlabel('Z slice')
#plt.ylabel('Number of beads')
#plt.show()

coor_3d = np.concatenate((coor, np.expand_dims(z_max, axis=1)), axis=1)

# fun = functools.partial(error, points=coor_3d)
# p0 = [0,0,0]
# res = scipy.optimize.minimize(fun, p0, method='Nelder-Mead')
# beta = res.x

#A = np.array([coor_3d[:, 0], coor_3d[:, 1], np.ones_like(coor_3d[:, 0])]).T
#beta = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), np.expand_dims(coor_3d[:, 2], axis=1))

#point = np.array([0.0, 0.0, beta[2]])
#normal = np.array(np.cross([1, 0, beta[0][0]], [0, 1, beta[1][0]]))
#d = -point.dot(normal)
#xx, yy = np.meshgrid([0, 2048], [0, 2048])
#z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(coor_3d[:, 0], coor_3d[:, 1], coor_3d[:, 2])
#ax.plot_surface(xx, yy, z, alpha=0.2, color=[0, 1, 0])
#plt.show()
stop = time.time()
print('duration: ', stop-start)

fig = plt.figure()
plt.scatter(coor_3d[:,0], coor_3d[:,1], c=coor_3d[:,2])
plt.colorbar()
plt.show()

xx, yy = np.meshgrid([0, 2048], [0, 2048])

#no_trend = coor_3d[:, 2] - (beta[0] * coor_3d[:, 0] + beta[1] * coor_3d[:, 1] + beta[2])

#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(coor_3d[:, 0], coor_3d[:, 1], no_trend)
#plt.show()

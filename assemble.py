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

def assemble(my_path):
    #constants 
    step = 10

    #main
    f = mrc.open(my_path,'r',permissive=True)
    h = f.header
    data = f.data[:,::10,::10]
    dimensions=data.shape

    eh = np.frombuffer(f.extended_header,dtype='i2')
    #pg.show(data)

    pos_x = eh[1:step*dimensions[0]:step]//10
    pos_y = eh[2:step*dimensions[0]:step]//10
    pos_z = eh[3:step*dimensions[0]:step]

    #step_x = pos_x[0]
    #x_i = 1 
    #while pos_x[x_i] == step_x:
    #    x_i += 1
    #step_x = np.abs(pos_x[x_i]-pos_x[0])
    #step_y = pos_y[0]
    #y_i = 1 
    #while y_i == pos_y[0]:
    #    y_i += 1
    #step_y = np.abs(pos_y[y_i]-pos_y[0])

    #x_max = np.max(pos_x)
    #y_max = np.max(pos_y)
    #print(step_x,step_y,x_max,y_max)

    #counter = 0
    #white_img = np.zeros_like(dimensions[1:2])
    #f, axarr = plt.subplots(x_max//step_x+1,y_max//step_y+1)
    #for counter in range(dimensions[0]):
    #    x = pos_x[counter]//step_x
    #    y = pos_y[counter]//step_y
    #    #print(x,y)
    #    axarr[x,y].imshow(data[counter],cmap='gray')
    #
    #plt.show()

    #cell_size_x = int(h['cella']['x'])
    #cell_size_y = int(h['cella']['y'])
    #cell_size_z = int(h['cella']['z'])

    #space_x = cell_size_x/dimensions[2]
    #space_y = cell_size_y/dimensions[1]
    #space_z = cell_size_z/dimensions[0]

    #stage_x = eh[4:step*dimensions[0]:step]/25 #stage position in microns
    #stage_y = eh[5:step*dimensions[0]:step]/25

    #stage_y_flipped = np.flip(stage_y,axis=0)
    #stage_y_offset = (stage_y_flipped+np.max(stage_y_flipped))/100
    #stage_y_scaled = stage_y_offset*10**6/space_y
    #stage_y_scaled = np.array([int(i) for i in stage_y_scaled])

    #stage_x_flipped = np.flip(stage_x,axis=0)
    #stage_x_offset = (stage_x_flipped+np.max(stage_x_flipped))/100
    #stage_x_scaled = stage_x_offset*10**6/space_y
    #stage_x_scaled = np.array([int(i) for i in stage_x_scaled])

    cy, cx = np.indices(dimensions[1:3])

    merged = np.zeros((np.max(pos_x)+dimensions[2],np.max(pos_y)+dimensions[1]))
    counts = np.zeros_like(merged)
    for i in range(dimensions[0]):
    #for i in range(5):
        print('Merge for image {}'.format(i))
        np.add.at(counts,(cx+pos_x[i],cy+pos_y[i]),1)
        np.add.at(merged,(cx+pos_x[i],cy+pos_y[i]),data[i])

    merged[counts>0] /= counts[counts>0]

    #pos_x_stage = stage_x_scaled
    #pos_y_stage = stage_y_scaled
    #merged = np.zeros((np.max(pos_x_stage)+dimensions[2],np.max(pos_y_stage)+dimensions[1]))
    #counts = np.zeros_like(merged)
    #for i in range(dimensions[0]):
    #   print('Merge for image {}'.format(i))
    #   np.add.at(counts,(cx+pos_x_stage[i],cy+pos_y_stage[i]),1)
    #   np.add.at(merged,(cx+pos_x_stage[i],cy+pos_y_stage[i]),data[i])

    #merged[counts>0] /= counts[counts>0]
    #np.save('merged.npy',merged)

    #plt.figure()
    #plt.imshow(np.abs(merged),vmin=0,vmax=0.01,cmap='gray')
    #plt.colorbar()
    #plt.imsave('assembled_image.tif',merged)
    #plt.show()

    return merged

if __name__=='__main__':
    path = '../gs.mrc'
    merged = assemble(path)
    #img = Image.fromarray(merged)
    #img.save('assembled.tif')
    pg.show(merged.T)















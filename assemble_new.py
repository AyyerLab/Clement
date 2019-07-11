#import cv2
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
plt.ion()
#constants
step = 10

#main
f = mrc.open('../gs.mrc','r',permissive=True)
h = f.header
data = f.data
dimensions=data.shape

eh = np.frombuffer(f.extended_header,dtype='i2')
#pg.show(data)

pos_x = eh[1:step*dimensions[0]:step]
pos_y = eh[2:step*dimensions[0]:step]
pos_z = eh[3:step*dimensions[0]:step]

step_x = pos_x[0]
x_i = 1
while pos_x[x_i] == step_x:
    x_i += 1
step_x = np.abs(pos_x[x_i]-pos_x[0])
step_y = pos_y[0]
y_i = 1
while y_i == pos_y[0]:
    y_i += 1
step_y = np.abs(pos_y[y_i]-pos_y[0])

#offset = np.array((dimensions[1],dimensions[2]))
cx,cy = np.indices(dimensions[1:3])
tmp = np.copy(pos_y)
pos_y = np.array(pos_x) #+ offset[0]
pos_x = np.array(tmp) #+ offset[1]
x_max = np.max(pos_x)
y_max = np.max(pos_y)
y_offset = 1500
x_offset = 1500
num_rows = len(set(pos_y))
counter = 0 
total_img = np.zeros((np.max(pos_x)+dimensions[2]+2000,np.max(pos_y)+dimensions[1]+2000))
counts = np.zeros_like(total_img)
shift = np.zeros((2,dimensions[0])).astype(np.int16)
new_positions_x = []
new_positions_y = []
for row  in range(num_rows):
    start = counter
    merged = np.zeros((x_max+10000,10000))
    mask = np.zeros_like(merged)
    while pos_y[counter] == pos_y[start]:
        print('Row: ',row)
        #pg.show(data[i-1])
         #pg.show(data[i])
        print('pos_x,pos_y: ',pos_x[counter],pos_y[counter])
        A = pos_x[counter]
        B = pos_x[counter] + dimensions[1] 
        C = y_offset
        D = y_offset + dimensions[2]
        ref_img = np.copy(merged[A:B,C:D])
        mask_i = np.copy(mask[A:B,C:D])
        help_img = mask_i * data[counter]
        ref_fft = np.fft.fft2(ref_img)
        data_fft = np.fft.fft2(help_img)
        corr_new = np.fft.fftshift(np.fft.ifft2(ref_fft.conj()*data_fft))
        my_max = np.array(np.unravel_index(np.argmax(corr_new),ref_img.shape))
        
        shift[0,counter] = int(dimensions[1]//2-my_max[0])
        shift[1,counter] = int(dimensions[2]//2-my_max[1])
        if counter==start:
            shift[:,start] = 0
        print(shift[0,counter],shift[1,counter])
        if np.abs(shift[0,counter])>1000:
            shift[0,counter]=shift[0,counter-1]
        if np.abs(shift[1,counter])>1000:
            shift[1]=shift[1,counter-1]
        print(shift[0,counter],shift[1,counter])
        np.add.at(merged,(cx+pos_x[counter]+shift[0,counter],cy+y_offset+shift[1,counter]),data[counter])
        np.add.at(mask,(cx+pos_x[counter]+shift[0,counter],cy+y_offset+shift[1,counter]),1)
        merged[mask==2] /= mask[mask==2]
        mask[mask==2] /= mask[mask==2]
        np.add.at(counts,(cx+pos_x[counter]+shift[0,counter],cy+pos_y[counter]+shift[1,counter]),1)
        np.add.at(total_img,(cx+pos_x[counter]+shift[0,counter],cy+pos_y[counter]+shift[1,counter]),data[counter])    
        new_positions_x.append(pos_x[counter]+shift[0,counter])
        new_positions_x.append(pos_y[counter]+shift[1,counter])
        counter += 1
        if counter == len(pos_y):
            break
    pg.show(merged[::10,::10])
total_img[counts>0] /= counts[counts>0]
pg.show(total_img[::10,::10])



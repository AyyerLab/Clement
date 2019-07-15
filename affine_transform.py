import numpy as np
from skimage import transform as tf
from skimage import data
from matplotlib import pyplot as plt
import scipy.ndimage as nd

def calc_dest_points(my_points):
    
    print(my_points)
    side_list = []
    for i in range(1,my_points.shape[0]):
        side_list.append(np.linalg.norm(my_points[i]-my_points[i-1]))
    side_list.append(np.linalg.norm(my_points[0]-my_points[3]))

    side_length = np.mean(side_list)
    print('ROI side length: ',side_length)
    print('ROI side length std: ', np.std(side_list))
    
    new_points = np.zeros_like(my_points)
    new_points[0] = my_points[0]
    new_points[1] = my_points[0] + (side_length,0)
    new_points[2] = my_points[0] + (side_length,side_length)
    new_points[3] = my_points[0] + (0,side_length)
    
    return new_points


def calc_affine_transform(my_points,new_points):
    
    return tf.estimate_transform('affine',my_points,new_points).params


def perform_affine_transform(my_points,matrix,image):
    
    pad_range = 256
    img_padded = np.pad(image,((pad_range,pad_range),(pad_range,pad_range)),mode='constant',constant_values=((0,0),(0,0)))
    
    #print(matrix)
    #print(image.shape)
    #center = 0.5*np.array((image.shape[0],image.shape[1],0)).reshape(1,3)
    #offset = (center + np.dot(center,matrix))
    #print(offset)

    #return tf.warp(img_padded,matrix,mode='wrap',preserve_range=True)
    return nd.affine_transform(img_padded,matrix)#,offset=offset)

if __name__ == '__main__':

    points = np.zeros((4,2))
    points[0] = (0,0)
    points[1] = (2,2)
    points[2] = (0,4)
    points[3] = (-2,2)

    pad_range = 512

    img1 = np.rot90(data.camera())
    pts2 = calc_dest_points(points)

    transform = calc_affine_transform(points,pts2)

    img2 = perform_affine_transform(points,transform,img1)
    
    plt.figure()
    plt.imshow(img1)
    plt.figure()
    plt.imshow(img2)
    plt.show()

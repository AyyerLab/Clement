import numpy as np
from skimage import transform as tf
from skimage import data
from matplotlib import pyplot as plt
import scipy.ndimage as nd
from PIL import Image
from matplotlib import patches

def calc_dest_points(my_points):
    print('Input points:\n', my_points)
    side_list = np.linalg.norm(np.diff(my_points, axis=0), axis=1)
    side_list = np.append(side_list, np.linalg.norm(my_points[0] - my_points[-1]))

    side_length = np.mean(side_list)
    print('ROI side length:', side_length, '\xb1', side_list.std())
    
    cen = my_points.mean(0) - np.ones(2)*side_length/2.
    new_points = np.zeros_like(my_points)
    new_points[0] = cen + (0, 0)
    new_points[1] = cen + (side_length,0)
    new_points[2] = cen + (side_length,side_length)
    new_points[3] = cen + (0,side_length)
    
    return new_points

def calc_affine_transform(my_points, new_points):
    return tf.estimate_transform('affine', my_points[:4], new_points).params

def perform_affine_transform(my_points, matrix, image, return_shift=False):
    nx, ny = image.shape
    corners = np.array([[0,0,1], [nx,0,1], [nx,ny,1], [0,ny,1]]).T
    tr_corners = np.dot(matrix, corners)
    output_shape = tuple([int(i) for i in (tr_corners.max(1) - tr_corners.min(1))[:2]])
    matrix[:2,2] -= tr_corners.min(1)[:2]
    print('Transform matrix:\n', matrix)

    trimage = nd.affine_transform(image, np.linalg.inv(matrix), order=1, output_shape=output_shape)
    if return_shift:
        return trimage, -tr_corners.min(1)[:2]
    else:
        return trimage

if __name__ == '__main__':
    points = np.array([[564.9,-42.6], [1249.1,689.0], [537.5, 1356.5], [-125.5, 569.9]])
    img1 = np.array(Image.open('c06_red.tif'))
    #points = np.zeros((4,2))
    #points[0] = (0,0)
    #points[1] = (2,2)
    #points[2] = (0,4)
    #points[3] = (-2,2)
    #img1 = np.rot90(data.camera())

    pts2 = calc_dest_points(points)

    transform = calc_affine_transform(points, pts2)

    img2, shift = perform_affine_transform(points, transform, img1, return_shift=True)

    plt.figure(figsize=(12,6))
    
    plt.subplot(121)
    plt.imshow(img1, vmax=800, cmap='gray')
    poly = patches.Polygon(points[:,::-1], ec='lime', fc='None')
    plt.gca().add_patch(poly)

    plt.subplot(122)
    plt.imshow(img2, vmax=800, cmap='gray')
    poly = patches.Polygon((pts2+shift)[:,::-1], ec='lime', fc='None')
    plt.gca().add_patch(poly)

    plt.show()

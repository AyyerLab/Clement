import read_lif
from peak_finding import Peak_finding
import numpy as np
from scipy import ndimage as ndi
from matplotlib import pyplot as plot
import pyqtgraph as pg
from PyQt5 import QtCore
from scipy.optimize import curve_fit
import numpy.ma as ma

threshold = 5
plt = 10
put = 200
flood_steps = 10
roi_size = 10
correct = False
sigma = 10


def subtract_background(img, sigma=None):
    if sigma is None:
        sigma = sigma_background
    norm = img.max()
    img_blurred = ndi.gaussian_filter(img, sigma=sigma)
    diff = img - img_blurred
    diff /= diff.max()
    return diff * norm


def peak_finding(im, threshold, plt, put, flood_steps, roi_size, background_correction=False, sigma_background=5):
    if background_correction:
        img = subtract_background(im, sigma_background)
    else:
        img = np.copy(im)
    img[img < threshold] = 0

    labels, num_objects = ndi.label(img)
    label_size = np.bincount(labels.ravel())

    # single photons and no noise
    mask_sp = np.where((label_size >= plt) & (label_size < put), True, False)
    if sum(mask_sp) == 0:
        coor_sp = []
    else:
        label_mask_sp = mask_sp[labels.ravel()].reshape(labels.shape)
        labels_sp = label_mask_sp * labels
        labels_sp, n_s = ndi.label(labels_sp)
        coor_sp = ndi.center_of_mass(img, labels_sp, range(1, labels_sp.max() + 1))

    # multiple photons
    mask_mp = np.where((label_size >= put) & (label_size < np.max(label_size)), True, False)
    if sum(mask_mp) > 0:
        label_mask_mp = mask_mp[labels.ravel()].reshape(labels.shape)
        labels_mp = label_mask_mp * labels
        labels_mp, n_m = ndi.label(labels_mp)
        for i in range(1, sum(mask_mp) + 1):
            slice_x, slice_y = ndi.find_objects(labels_mp == i)[0]
            roi_i = np.copy(img[slice_x, slice_y])
            max_i = np.max(roi_i)
            step = (0.95 * max_i - threshold) / flood_steps
            multiple = False
            coor_tmp = np.array(ndi.center_of_mass(roi_i, ndi.label(roi_i)[0]))
            for k in range(1, flood_steps + 1):
                new_threshold = threshold + k * step
                roi_i[roi_i < new_threshold] = 0
                labels_roi, n_i = ndi.label(roi_i)
                if n_i > 1:
                    roi_label_size = np.bincount(labels_roi.ravel())
                    if np.max(roi_label_size[1:]) <= put:
                        if len(roi_label_size) == 3 and roi_label_size.min() < roi_min_size:
                            break
                        else:
                            multiple = True
                            coordinates_roi = np.array(ndi.center_of_mass(roi_i, labels_roi, range(1, n_i + 1)))
                            [coor_sp.append(coordinates_roi[j] + np.array((slice_x.start, slice_y.start))) for j
                             in
                             range(len(coordinates_roi))]
                            break
            if not multiple:
                coor_sp.append(coor_tmp + np.array((slice_x.start, slice_y.start)))

    coor = np.array(coor_sp)
    peaks_2d = np.round(coor)
    print('Number of peaks found: ', peaks_2d.shape[0])
    return peaks_2d


fname = '../data/PSF_warm_and_stack-series/zstacks_20201102.lif'
base_reader = read_lif.Reader(fname)

shifted_mean = []
psf_2d = []
all_profiles = []
mu_list = []
coor = []
sigma_fix = None
pixel_size = [128, 477]
psize = [1,1]
std_list = []
for k in range(1,3):
    reader = base_reader.getSeries()[k]
    data = reader.getFrame(channel=0, dtype='u2').astype('f8')
    data = (data - data.min()) / (data.max() - data.min())
    data *= 100
    md = reader.getMetadata()
    psize[k-1] = np.array((md['voxel_size_x'], md['voxel_size_y'], md['voxel_size_z']))
    max_proj = data.max(axis=0)
    max_int = np.max(data)
    mean_int = max_int
    peaks = peak_finding(max_proj, threshold, plt, put, flood_steps, roi_size, correct, sigma)
    coor.append(peaks)
    size = 10
    #fig = pg.show(max_proj)
    #for i in range(peaks.shape[0]):
    #    pos = QtCore.QPointF(peaks[i][0] - size / 2, peaks[i][1] - size / 2)
    #    point = pg.CircleROI(pos, size, parent=fig.getImageItem(),
    #                               movable=True, removable=True)
    #    point.setPen(0, 255, 255)
    #    point.removeHandle(0)
    #    fig.addItem(point)


    x_red = np.arange(data.shape[0])
    gauss = lambda x, mu, sigma, offset: mean_int * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + offset
    mu = []
    argmax = []
    std = []
    fits = []

    mean_sigma = None
    sigma_list = []
    rois = []
    pillars = []
    figures = []
    z_profiles = []
    for i in range(len(peaks)):
        #z_profile = data[:,np.round(peaks[i,0]).astype(int), np.round(peaks[i,1]).astype(int)]
        z_profile = data[:,np.round(coor[0][i,0]).astype(int), np.round(coor[0][i,1]).astype(int)]
        z_profiles.append(z_profile)

        z_peak = x_red[z_profiles[i] > np.exp(-0.5) * z_profiles[i].max()]
        sigma_guess = 0.5 * (z_peak.max() - z_peak.min())
        if sigma_guess == 0:
            plot.figure()
            plot.plot(x_red, z_profiles[i])
            sigma_guess = 0.5
        offset = z_profiles[i].min()
        mask = np.zeros_like(z_profiles[i]).astype(int)
        mask[z_profiles[i] == max_int] = 1
        x_masked = ma.masked_array(x_red, mask)
        z_masked = ma.masked_array(z_profiles[i], mask)
        popt, pcov = curve_fit(gauss, x_masked, z_masked, p0=[np.argmax(z_profiles[i]), sigma_guess, offset])
        if popt[0] > 0 and popt[0] < z_profile.shape[0]:
            sigma_list.append(popt[1])

        a = np.round(peaks[i,0]-roi_size).astype(int)
        b = np.round(peaks[i,0]+roi_size).astype(int)
        c = np.round(peaks[i,1]-roi_size).astype(int)
        d = np.round(peaks[i,1]+roi_size).astype(int)

        roi = max_proj[a:b, c:d]
        pillar = data[:, a:b, c:d]
        if roi.shape == (size*2, size*2):
            rois.append(roi)
        if pillar.shape[1:] == (size*2, size*2):
            pillars.append(pillar)

    mean_sigma = np.median(sigma_list)
    if sigma_fix is None:
        sigma_fix = mean_sigma * pixel_size[0]

    #gauss_static = lambda x, mu, offset: mean_int * np.exp(-(x - mu) ** 2 / (2 * mean_sigma ** 2)) + offset
    gauss_static = lambda x, mu, offset: mean_int * np.exp(-(x - mu) ** 2 / (2 * (sigma_fix/pixel_size[k-1]) ** 2)) + offset
    print('Mean sigma: ', mean_sigma)
    profiles_shifted = []
    for i in range(len(z_profiles)):
        try:
            mask = np.zeros_like(z_profiles[i]).astype(int)
            mask[z_profiles[i] == max_int] = 1
            x_masked = ma.masked_array(x_red, mask)
            z_masked = ma.masked_array(z_profiles[i], mask)
            popt, pcov = curve_fit(gauss_static, x_masked, z_masked, p0=[np.argmax(z_profiles[i]), offset])
            fit_i = gauss_static(x_red, popt[0], popt[1])
            fits.append(fit_i)
            perr = np.sqrt(np.diag(pcov))
            mu.append(popt[0])
            std.append(perr[0])
            argmax.append(np.argmax(z_profile[i]))

            shift = z_profiles[i].shape[0] - popt[0]
            shifted = np.zeros(2*z_profiles[i].shape[0])
            start = shift
            stop = start + z_profiles[i].shape[0]
            shifted[int(np.round(start)):int(np.round(stop))] = z_profiles[i]
            profiles_shifted.append(shifted)

            #if k == 2:
            #    plot.figure()
            #    plot.plot(np.arange(z_profiles[i].shape[0]), z_profiles[i])
            #    plot.plot(np.arange(z_profiles[i].shape[0]), fits[i])

        except RuntimeError:
            mu.append(0)
            argmax.append(0)
            print('Runtime error for profile: ', i)
            plot.figure()
            plot.plot(np.arange(z_profiles[i].shape[0]), z_profiles[i])
            #plot.plot(np.arange(z_profiles[i].shape[0]), fits[i])

    psf_2d.append(np.array(rois).mean(0))
    shifted_mean.append(np.array(profiles_shifted).mean(0))
    all_profiles.append(z_profiles)
    mu_list.append(mu)
    std_list.append(std)

x1 = np.arange(len(shifted_mean[0]))
x2 = np.arange(len(shifted_mean[1]))

shift_again = np.argmax(shifted_mean[0])*130 - np.argmax(shifted_mean[1])*500
x2 *= 500
x2 += shift_again

#plot.figure()
#plot.plot(x1*130, shifted_mean[0])
#plot.plot(x2, shifted_mean[1])
#plot.show()


#plot.figure()
#plot.show()

mu_normed = [np.array(mu_list[0])*130, np.array(mu_list[1])*500]
mu_diff = np.array(mu_list[0]) * 130 - np.array(mu_list[1]) * 500

all_maxima = []
for k in range(2):
    maxima = []
    for i in range(len(all_profiles[k])):
        argmax = np.array(all_profiles[k][i]).argmax()
        if k == 1:
            argmax = len(all_profiles[1][0]) - argmax
        maxima.append(argmax)
    all_maxima.append(maxima)

new_diff = np.array(all_maxima[0]) * 130 - np.array(all_maxima[1])*500
print(new_diff)


#plot.figure()
#plot.scatter(coor[0][:,0], coor[0][:,1],c=mu_diff)
#plot.colorbar()
#plot.figure()
#plot.scatter(coor[1][:,0], coor[1][:,1],c=mu_diff)
#plot.colorbar()
#plot.show()

#p = coor[0]
#p_prime = coor[1]
#Q = p[1:] - p[0]
#Q_prime = p_prime[1:] - p_prime[0]

#R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
#           np.row_stack((Q_prime, np.cross(*Q_prime))))

#t = p_prime[0] - np.dot(p[0], R)

#A = np.column_stack((np.row_stack((R,t)),
#                     (0,0,0,1)))


p = []
p_prime = []
for i in range(len(coor[0])):
#for i in range(6):
    point = [coor[0][i,0], coor[0][i,1], mu_list[0][i], 1]
    p.append(point)
    point = [coor[1][i,0], coor[1][i,1], mu_list[1][i], 1]
    p_prime.append(point)

p = np.array(p)
p_prime = np.array(p_prime)
p[:,2] *= 130
p_prime[:,2] *= 500
#Q = p[1:] - p[0]
#Q_prime = p_prime[1:] - p_prime[0]

#R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
#          np.row_stack((Q_prime, np.cross(*Q_prime))))

#t = p_prime[0] - np.dot(p[0], R)

#A = np.column_stack((np.row_stack((R,t)),
#                     (0,0,0,1)))

A = p_prime @ p.T @ np.linalg.inv(p @ p.T)


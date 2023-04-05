import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import curve_fit
import time
from skimage import measure, morphology
import read_lif
import copy

class Peak_finding():
    def __init__(self, threshold=0, plt=10, put=200):
        self.num_slices = None
        self.peaks_orig = None
        self.peaks = None
        self.tf_peaks = None
        self.orig_tf_peaks = None
        self.tf_peaks_z = None
        self.peaks_z = None
        self.peaks_align_ref = None
        self.tf_peaks_align_ref = None
        self.pixel_lower_threshold = plt
        self.pixel_upper_threshold = put
        self.flood_steps = 10
        self.threshold = threshold
        self.sigma_background = 5
        self.roi_min_size = 10
        self.background_correction = False
        self.adjusted_params = False
        self.sigma_z = None
        self.aligning = False
        self.my_counter = None


    def peak_finding(self, im, transformed, roi=False, roi_pos=None, background_correction=None):
        start = time.time()
        img = np.copy(im)
        #if background_correction is None and self.background_correction:
        #    img = self.subtract_background(img)

        self.log(self.threshold)
        self.log(self.pixel_upper_threshold)
        self.log(self.pixel_lower_threshold)
        self.log(self.flood_steps)
        self.log(img.shape)
        self.log(img.max())
        #img[img < self.threshold] = 0

        labels, num_objects = ndi.label(img)
        label_size = np.bincount(labels.ravel())

        # single photons and no noise
        mask_sp = np.where((label_size >= self.pixel_lower_threshold) & (label_size < self.pixel_upper_threshold),
                           True, False)
        if sum(mask_sp) == 0:
            coor_sp = []
        else:
            label_mask_sp = mask_sp[labels.ravel()].reshape(labels.shape)
            labels_sp = label_mask_sp * labels
            labels_sp, n_s = ndi.label(labels_sp)
            coor_sp = ndi.center_of_mass(img, labels_sp, range(1, labels_sp.max() + 1))

        # multiple photons
        mask_mp = np.where((label_size >= self.pixel_upper_threshold) & (label_size < np.max(label_size)), True,
                           False)
        if sum(mask_mp) > 0:
            label_mask_mp = mask_mp[labels.ravel()].reshape(labels.shape)
            labels_mp = label_mask_mp * labels
            labels_mp, n_m = ndi.label(labels_mp)
            for i in range(1, sum(mask_mp) + 1):
                objects = ndi.find_objects(labels_mp == i)
                if len(objects) == 0:
                    self.print('No beads found!')
                    return None
                slice_x, slice_y = objects[0]
                roi_i = np.copy(img[slice_x, slice_y])
                max_i = np.max(roi_i)
                step = (0.95 * max_i - self.threshold) / self.flood_steps
                multiple = False
                coor_tmp = np.array(ndi.center_of_mass(roi_i, ndi.label(roi_i)[0]))
                for k in range(1, self.flood_steps + 1):
                    new_threshold = self.threshold + k * step
                    roi_i[roi_i < new_threshold] = 0
                    labels_roi, n_i = ndi.label(roi_i)
                    if n_i > 1:
                        roi_label_size = np.bincount(labels_roi.ravel())
                        if np.max(roi_label_size[1:]) <= self.pixel_upper_threshold:
                            if len(roi_label_size) == 3 and roi_label_size.min() < self.roi_min_size:
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
        if len(coor) == 0:
            return None
        if roi:
            try:
                return np.round(coor)[0]
            except IndexError:
                return None
        else:
            #peaks_2d = np.round(coor)
            peaks_2d = coor
            if roi_pos is not None and peaks_2d is not None:
                peaks_2d += roi_pos
            if self.aligning:
                if transformed:
                    self.tf_peaks_align_ref = np.copy(peaks_2d)
                else:
                    self.peaks_align_ref = np.copy(peaks_2d)
            else:
                if transformed:
                    self.tf_peaks = peaks_2d
                    if self.orig_tf_peaks is None:
                        self.orig_tf_peaks = np.copy(self.tf_peaks)
                else:
                    self.peaks_orig = np.copy(peaks_2d)
                    self.peaks = np.copy(self.peaks_orig)
        end = time.time()
        self.log('duration: ', end - start)
        self.print('Number of peaks found: ', peaks_2d.shape[0])
        #self.load_channel(0)
        #self.peaks_z = []
        #for i in range(len(self.peaks_orig)):
        #    a, b, c = self.gauss_3d(self.peaks_orig[i], False, channel=0, slice=None, size=10)
        #    self.peaks_orig[i] = a[:2]
        #    self.peaks_z.append(a[-1])
        #self.peaks = np.copy(self.peaks_orig)

    def update_peaks(self, tf_matrix, transformed):
        if transformed:
            for i in range(len(self.peaks)):
                self.peaks[i][:2] = (tf_matrix @ np.array([self.peaks_orig[i, 0], self.peaks_orig[i, 1], 1]))[:2]
            self.tf_peaks = np.copy(self.peaks)
        else:
            self.peaks = np.copy(self.peaks_orig)

    def subtract_background(self, img, sigma=None):
        #if sigma is None:
        #    sigma = self.sigma_background
        #norm = img.max()
        #img_blurred = ndi.gaussian_filter(img, sigma=sigma)
        #diff = img-img_blurred
        #diff /= diff.max()/norm
        img_sub = img - np.median(img)
        #return diff - np.median(diff)
        return img_sub

    def calc_original_coordinates(self, tf_mat, point=None):
        if point is not None:
            point = np.array([point[0], point[1], 1])
            nx, ny = self.orig_data.shape[:-1]
            corners = np.array([[0, 0, 1], [nx, 0, 1], [nx, ny, 1], [0, ny, 1]]).T
            tf_corners = np.dot(tf_mat, corners)
            tf_matrix = np.copy(tf_mat)
            tf_matrix[:2,2] += tf_corners.min(1)[:2]
            return (np.linalg.inv(tf_mat) @ point)[:2]
        else:
            inv_points = []
            for i in range(len(self.points)):
                inv_points.append((np.linalg.inv(tf_mat) @ np.array([self.points[i,0], self.points[i,1], 1]))[:2])
            return inv_points


    def fit_z(self, data, local=False, point=None):
        '''
        calculates the z profile along the beads and fits a gaussian
        '''
        if not local:
            peaks_2d = self.peaks_orig
        else:
            peaks_2d = point

        if peaks_2d is None:
            if local:
                self.print('You have to parse a point!')
            else:
                self.print('Calculate 2d peaks first!')
            return

        z_profile = data[np.round(peaks_2d[:, 0]).astype(int), np.round(peaks_2d[:, 1]).astype(int)]
        mean_int = np.median(np.max(z_profile, axis=1), axis=0)
        max_int = np.max(z_profile)
        x = np.arange(z_profile.shape[1])

        gauss = lambda x, mu, sigma, offset: mean_int * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + offset

        sigma_list = []
        for i in range(len(z_profile)):
            try:
                z_peak = x[z_profile[i] > np.exp(-0.5) * z_profile[i].max()]
                sigma_guess = 0.5 * (z_peak.max() - z_peak.min())
                if sigma_guess == 0:
                    sigma_guess = 2
                offset = z_profile[i].min()
                mask = np.zeros_like(z_profile[i]).astype(int)
                mask[z_profile[i] == max_int] = 1
                x_masked = np.ma.masked_array(x, mask)
                z_masked = np.ma.masked_array(z_profile[i], mask)
                popt, pcov = curve_fit(gauss, x_masked, z_masked, p0=[np.argmax(z_profile[i]), sigma_guess, offset])
                if popt[0] > 0 and popt[0] < z_profile.shape[1]:
                    sigma_list.append(popt[1])
            except RuntimeError:
                pass
        if local:
            return popt[0]
        else:
            if len(sigma_list) == 0:
                self.print('Z fitting of the beads failed. Contact developers!')
                return
            self.sigma_z = np.median(sigma_list)
            gauss_static = lambda x, mu, offset: mean_int * np.exp(-(x - mu) ** 2 / (2 * self.sigma_z ** 2)) + offset
            mean_values = []
            for i in range(len(z_profile)):
                try:
                    mask = np.zeros_like(z_profile[i]).astype(int)
                    mask[z_profile[i] == max_int] = 1
                    x_masked = np.ma.masked_array(x, mask)
                    z_masked = np.ma.masked_array(z_profile[i], mask)
                    popt, pcov = curve_fit(gauss_static, x_masked, z_masked, p0=[np.argmax(z_profile[i]), offset])
                    perr = np.sqrt(np.diag(pcov))
                except RuntimeError:
                    self.print('Runtime error for profile: ', i)
                    self.print('Unable to fit z profile. Calculate argmax(z).')
                    self.print('WARNING! Calculation of the z-position might be inaccurate!')
                    mean_values.append(np.argmax(z_profile[i]))
                mean_values.append(popt[0])

            self.peaks_z = np.copy(mean_values)

    def calc_local_z(self, data, point, transformed=True, tf_matrix=None):
        if transformed:
            point = self.calc_original_coordinates(point, tf_matrix)
        z = None
        try:
            if point[0] < 0 or point[1] < 0:
                raise IndexError
            point = np.expand_dims(point, axis=0)
            z = self.fit_z(data,  local=True, point=point)
        except IndexError:
            self.print('You should select a point within the bounds of the image!')
        #finally:
        return z

    def check_peak_index(self, point, size):
        diff = self.peaks - point
        diff_err = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
        ind_arr = np.where(diff_err < size / 2)[0]
        if len(ind_arr) == 0:
            return None
        elif len(ind_arr) > 1:
            self.print('Selection ambiguous. Try again!')
        else:
            return ind_arr[0]

    def gauss_3d(self, point, transformed, channel=None, slice=None, size=10):
        def fit_func(mesh, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, intens, offset):
            x, y, z = mesh
            return (intens * np.exp(-(x - mu_x) ** 2 / (2 * sigma_x **2)) * np.exp(-(y - mu_y) ** 2 / (2 * sigma_y ** 2)) *
                    np.exp(-(z - mu_z) ** 2 / (2 * sigma_z ** 2)) + offset).ravel()

        if channel is None:
            channel = self._channel_idx

        if transformed:
            tf_aligned = self.tf_matrix @ self._color_matrices[channel]
            point = self.calc_original_coordinates(tf_aligned, point)
        else:
            point = np.linalg.inv(self._color_matrices[channel]) @ np.array([point[0], point[1], 1])


        if point[0] < 0 or point[1] < 0 or point[0] > self.orig_data.shape[0] or point[1] > self.orig_data.shape[1]:
            self.print('You have to select a point within the bounds of the image!')
            return None, None, None

        idx = None
        if self.peaks is not None:
            idx = self.check_peak_index(point[:2], size)
        point_tmp = np.copy(point)
        if idx is not None:
            peaks_2d = self.peaks[idx]
            bead = True
        else:
            peaks_2d = point_tmp
            bead = False

        roi_size = 10
        x_min = np.round(point[0] - roi_size/2).astype(int)
        x_max = np.round(point[0] + roi_size/2).astype(int)
        y_min = np.round(point[1] - roi_size/2).astype(int)
        y_max = np.round(point[1] + roi_size/2).astype(int)

        x0, y0 = (roi_size/2, roi_size/2)
        data = self.channel[x_min:x_max, y_min:y_max, :]
        if slice is None:
            z0 = np.argmax(data[np.round(x0).astype(int), np.round(y0).astype(int), :])
        else:
            z0 = slice

        if self.my_counter is None:
            self.my_counter = 0
        self.my_counter += 1

        #np.save('point{}.npy'.format(self.my_counter), peaks_2d)

        offset = np.mean(data)
        max_proj = np.max(data, axis=-1)
        max_int = np.max(max_proj)
        max_shape = np.array(data.shape).max()

        #bounds for fitting params: x0, y0, z0, sigma_x, sigma_y, sigma_z, intensity, offset
        if bead:
            bounds = ((x0 - max_proj.shape[0] / 4, y0 - max_proj.shape[1] / 4, z0 - data.shape[-1] / 4, 0, 0, 0, np.mean(data), 0),
                     (x0 + max_proj.shape[0] / 4, y0 + max_proj.shape[1] / 4, z0 + data.shape[-1] / 4, 10, 10, 2, np.max(data), np.max(data)))
        else:
            bounds = ((x0 - max_proj.shape[0] / 4, y0 - max_proj.shape[1] / 4, z0 - data.shape[-1] / 4, 0, 0, 0, np.mean(data),np.mean(data)),
                      (x0 + max_proj.shape[0] / 4, y0 + max_proj.shape[1] / 4, z0 + data.shape[-1] / 4, 2, 2, 2, np.max(data), np.max(data)))

        x = np.arange(max_shape)
        x, y, z = np.meshgrid(x, x, x, indexing='ij')
        data_pad = np.zeros_like(x).astype('f8')
        data_pad[:data.shape[0], :data.shape[1], :data.shape[2]] = data

        try:
            popt, pcov = curve_fit(fit_func, (x, y, z), data_pad.ravel(), p0=[x0, y0, z0, 1, 1, 1, max_int, offset],
                                   bounds=bounds)
        except RuntimeError:
            self.print('Unable to fit virus! Select another one!')
            return None, None, None

        fit = fit_func((x, y, z), *popt).reshape(x.shape)[:data.shape[0], :data.shape[1], :data.shape[2]]
        ss_res = np.sum((data - fit) ** 2)
        ss_tot = np.sum((data - data.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        perr = np.sqrt(np.diag(pcov))
        point =  np.array([popt[0]+x_min, popt[1]+y_min, 1])

        z = popt[2]
        self.print(z*self.voxel_size*1e9)
        #np.save('z_values{}.npy'.format(self.my_counter), z)
        init = np.array([point[0], point[1], z])

        #np.save('poi{}_tb.npy'.format(self.my_counter), self.channel[np.rint(init[0]).astype(int), np.rint(init[1]).astype(int), :])
        self.log('Model fit: ', r2)

        if r2 < 0.2 and r2 > 0:
            self.print('Model does not fit well to the data. You should consider selecting a different virus!',
                       ' Uncertainty: ', perr[:3]*self.voxel_size*1e9)
        #elif r2 < 0:
        #    self.print('Unable to fit virus! Select another one!')
        else:
            self.print('Fitting succesful: ', init, ' Uncertainty: ', perr[:3]*self.voxel_size*1e9)

        return init, perr[:3], pcov[:3,:3]


    def reset_peaks(self):
        self.peaks = None
        self.tf_peaks = None
        self.orig_tf_peaks = None
        self.tf_peaks_z = None
        self.peaks_z = None
        self.mu = []
        self.adjusted_params = False

import os

import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
from operator import itemgetter
import yaml

class Project(QtWidgets.QWidget):
    def __init__(self, fm, em, fib, parent):
        super(Project,self).__init__()
        self._project_folder = None
        self.fm = fm
        self.em = em
        self.fib = fib
        self.show_fib = False
        self.merged = False
        self.popup = None
        self.parent = parent
        self.load_merge = False

    def _load_project(self):
        self._project_folder = os.getcwd()

        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                             'Select project',
                                                             self._project_folder,
                                                             '*.yml')
        if file_name is not '':
            self.fm.reset_base()
            self.fm.reset_init()
            self.em.reset_init()
            self.fib.reset_init()
            self._project_folder = os.path.dirname(file_name)
            with open(file_name, 'r') as f:
                project = yaml.load(f, Loader=yaml.FullLoader)
            self._load_fm(project)
            self._load_em(project)
            self._load_fib(project)
            self._load_base(project)

    def _load_fm(self, project):
        if 'FM' not in project:
            return
        fmdict = project['FM']
        self.fm._curr_folder = fmdict['Directory']
        self.fm._file_name = fmdict['File']
        self.fm._colors = fmdict['Colors']
        self.fm.c1_btn.setStyleSheet('background-color: {}'.format(self.fm._colors[0]))
        self.fm.c2_btn.setStyleSheet('background-color: {}'.format(self.fm._colors[1]))
        self.fm.c3_btn.setStyleSheet('background-color: {}'.format(self.fm._colors[2]))
        self.fm.c4_btn.setStyleSheet('background-color: {}'.format(self.fm._colors[3]))

        self.fm._channels = fmdict['Channels']
        self.fm.channel1_btn.setChecked(self.fm._channels[0])
        self.fm.channel2_btn.setChecked(self.fm._channels[1])
        self.fm.channel3_btn.setChecked(self.fm._channels[2])
        self.fm.channel4_btn.setChecked(self.fm._channels[3])
        self.fm.overlay_btn.setChecked(fmdict['Overlay'])

        if 'Series' in fmdict:
            self.fm._series = fmdict['Series']
        self.fm._parse_fm_images(self.fm._file_name, self.fm._series)

        if fmdict['Max projection orig']:
            self.fm.max_proj_btn.setChecked(True)
        else:
            self.fm.slice_select_btn.setValue(fmdict['Slice'])
            self.fm._slice_changed()

        if 'Align colors' in fmdict:
            self.fm.align_btn.setChecked(fmdict['Align colors'])

        try:
            self.fm.ops._orig_points = np.array(fmdict['Original grid points'])
            self.fm.ops.points = np.copy(self.fm.ops._orig_points)
            self.fm.show_grid_btn.setEnabled(True)
            self.fm._recalc_grid()
            self.fm.show_grid_btn.setChecked(fmdict['Show grid box'])
            self.fm.rot_transform_btn.setChecked(fmdict['Rotation only'])
            try:
                self.fm.ops._tf_points = np.array(fmdict['Transformed grid points'])
                self.fm._affine_transform(toggle_orig=False)
                if fmdict['Max projection transformed']:
                    self.fm.max_proj_btn.setChecked(True)
                else:
                    self.fm.slice_select_btn.setValue(fmdict['Slice'])
                    self.fm._slice_changed()
                self.fm.flipv.setChecked(fmdict['Flipv'])
                self.fm.fliph.setChecked(fmdict['Fliph'])
                self.fm.transpose.setChecked(fmdict['Transpose'])
                self.fm.rotate.setChecked(fmdict['Rotate'])
            except KeyError:
                pass
        except KeyError:
            pass

        if 'Show peaks' in fmdict:
            self.fm.peak_btn.setChecked(fmdict['Show peaks'])
        self.fm.show_btn.setChecked(fmdict['Show original'])
        self.fm.map_btn.setChecked(fmdict['Show z map'])
        self.fm.remove_tilt_btn.setChecked(fmdict['Remove tilt'])

        self.fm._refined = fmdict['Refined']
        if self.fm._refined:
            self.fm.ops._refine_shape = tuple(fmdict['Refine shape'])
            self.fm.ops._refine_matrix = np.array(fmdict['Refine matrix'])
            self.fm.ops.refine_history.append(self.fm.ops._refine_matrix)
            self.fm.ops.apply_refinement()

        self.fm._update_imview()

    def _load_em(self, project):
        if 'EM' not in project:
            return
        emdict = project['EM']
        self.em._curr_folder = emdict['Directory']
        self.em._file_name = emdict['File']
        self.em.mrc_fname.setText(self.em._file_name)
        self.em.assemble_btn.setEnabled(True)
        self.em.step_box.setEnabled(True)
        self.em.step_box.setText(emdict['Downsampling'])
        self.em._assemble_mrc()
        try:
            self.em._select_region_original = emdict['Select subregion original']
            try:
                self.em.ops._orig_points = np.array(emdict['Original grid points'])
                self.em.ops.points = np.copy(self.em.ops._orig_points)
                self.em.show_grid_btn.setEnabled(True)
                self.em._recalc_grid()
                self.em.show_grid_btn.setChecked(emdict['Show grid box'])
                self.em.rot_transform_btn.setChecked(emdict['Rotation only'])
            except KeyError:
                pass
            try:
                self.em.select_region_btn.setChecked(True)
                self.em._box_coordinate = np.array(emdict['Subregion coordinate'])
                self.em.select_region_btn.setChecked(False)
                self.em.ops._orig_points_region = np.array(emdict['Orginal points subregion'])
                self.em.ops.points = np.copy(self.em.ops._orig_points_region)
                self.em._recalc_grid()
                self.em.show_grid_btn.setEnabled(True)
                self.em.show_grid_btn.setChecked(emdict['Show grid box'])
                self.em.rot_transform_btn.setChecked(emdict['Rotation only'])
                try:
                    self.em.show_grid_btn.setChecked(False)
                    self.em.ops._tf_points_region = np.array(emdict['Transformed points subregion'])
                    self.em._affine_transform()
                except KeyError:
                    pass

                self.em.show_assembled_btn.setChecked(emdict['Show assembled'])
                if self.em.show_assembled_btn.isChecked():
                    try:
                        self.em.ops._tf_points = np.array(emdict['Transformed grid points'])
                        self.em._affine_transform()
                    except KeyError:
                        pass
            except KeyError:
                pass
        except KeyError:
            try:
                self.em.ops._orig_points = np.array(emdict['Original grid points'])
                self.em.ops.points = np.copy(self.em.ops._orig_points)
                self.em.show_grid_btn.setEnabled(True)
                self.em._recalc_grid()
                self.em.show_grid_btn.setChecked(emdict['Show grid box'])
                self.em.rot_transform_btn.setChecked(emdict['Rotation only'])
                try:
                    self.em.ops._tf_points = np.array(emdict['Transformed grid points'])
                    self.em._affine_transform()
                except KeyError:
                    pass
            except KeyError:
                pass
            try:
                self.em._box_coordinate = emdict['Subregion coordinate']
                self.em.select_region_btn.setChecked(True)
                self.em.select_region_btn.setChecked(False)
                self.em.ops._orig_points_region = np.array(emdict['Orginal points subregion'])
                self.em.ops.points = np.copy(self.em.ops._orig_points_region)
                self.em._recalc_grid()
                self.em.show_grid_btn.setChecked(emdict['Show grid box'])
                self.em.rot_transform_btn.setChecked(emdict['Rotation only'])
                try:
                    self.em.ops._tf_points_region = np.array(emdict['Transformed points subregion'])
                    self.em._affine_transform()
                except KeyError:
                    pass
            except KeyError:
                if self.em.select_region_btn.isChecked():
                    self.em.select_region_btn.setChecked(False)

        self.em.show_assembled_btn.setChecked(emdict['Show assembled'])
        self.em.show_btn.setChecked(emdict['Show original'])

    def _load_fib(self, project):
        if 'FIB' not in project:
            return
        fibdict = project['FIB']
        self.fib._curr_folder = fibdict['Directory']
        self.fib._file_name = fibdict['File']
        self.fib.mrc_fname.setText(self.fib._file_name)
        self.fib._load_mrc(jump=True)
        if fibdict['Transpose']:
            self.fib.transp_btn.setEnabled(True)
            self.fib.transp_btn.setChecked(True)
            self.fib._transpose() # Why has this function to be called expilicitely???

        self.fib.sigma_btn.setText(fibdict['Sigma angle'])
        self.fib.sem_ops = self.em.ops
        if self.fib.sem_ops._orig_points is not None:
            self.fib.enable_buttons(True)

        try:
            new_points = fibdict['Grid points']
            self.fib.show_grid_btn.setChecked(fibdict['Show grid'])
            self.fib.ops.points = np.copy(new_points)
            self.fib._calc_grid()
            #self.fib.ops.calc_grid_shift(new_points)
        except KeyError:
            pass

        if fibdict['Refined']:
            self.fib._refined = fibdict['Refined']
            self.fib.ops._refine_matrix = np.array(fibdict['Refine matrix'])
            self.fib.ops.apply_refinement(self.fib.ops.points)
            self.fib.ops.refine_grid()
            self.fib._calc_grid()

        self.parent.tabs.setCurrentIndex(fibdict['Tab index'])
        if fibdict['Tab index']:
            self.fib.show_fib = True
            self.show_fib = True

        if self.fib.ops.data is not None and fibdict['Tab index'] == 1:
            if self.fib.sem_ops is not None and self.fib.sem_ops._orig_points is not None:
                self.fib.show_grid_btn.setChecked(fibdict['Show grid'])
            self.fib.show_peaks_btn.setChecked(fibdict['Show peaks'])

    def _load_base(self, project):
        try:
            fmdict = project['FM']
            print('show fib: ', self.show_fib)
            if self.show_fib:
                self.fm.select_btn.setChecked(True)
                points_corr_fm = fmdict['Correlated points']
                qpoints = [QtCore.QPointF(p[0], p[1]) for p in np.array(points_corr_fm)]
                [self.fm._draw_correlated_points(point, self.fm.size_ops, self.fm.size_other, self.fm.imview.getImageItem())
                for point in qpoints]
                self.fm.select_btn.setChecked(False)

            else:
                emdict = project['EM']
                em = self.em
                self.fm.select_btn.setChecked(True)

                points_corr_fm = fmdict['Correlated points']
                indices_fm = fmdict['Correlated points indices']
                if len(indices_fm) > 0:
                    points_red = list(itemgetter(*indices_fm)(list(points_corr_fm)))
                    roi_list = [QtCore.QPointF(p[0],p[1]) for p in np.array(points_red)]
                    [self.fm._draw_correlated_points(roi, self.fm.size_ops, self.fm.size_other, self.fm.imview.getImageItem()) for roi in roi_list]
                if not self.show_fib:
                    try:
                        em.select_btn.setChecked(True)
                        points_corr_em = emdict['Correlated points']
                        indices_em = list(emdict['Correlated points indices'])
                        if len(indices_em) > 0:
                            points_red = list(itemgetter(*indices_em)(list(points_corr_em)))
                            roi_list = [QtCore.QPointF(p[0],p[1]) for p in np.array(points_red)]
                            [em._draw_correlated_points(roi, em.size_ops, em.size_other, em.imview.getImageItem()) for roi in roi_list]
                    except KeyError:
                        pass
                    em.select_btn.setChecked(False)
                self.fm.select_btn.setChecked(False)

                #### update/correct points because in draw_correlated_points the unmoved points are drawn in other.imview
                try: #do this only when correlated points exist
                    indices_fm = list(fmdict['Correlated points indices'])
                    if len(indices_fm) > 0:
                        correct_em = list(itemgetter(*indices_fm)(list(points_corr_em)))
                        pt_list_em = [QtCore.QPointF(p[0],p[1]) for p in np.array(correct_em)]
                        roi_list_em = [pg.CircleROI(pt_list_em[i],em.size_ops, parent=em.imview.getImageItem(), movable=True, removable=True) for i in range(len(pt_list_em))]
                        [roi.setPen(0,255,255) for roi in roi_list_em]
                        [roi.removeHandle(0) for roi in roi_list_em]
                        #[em.imview.removeItem(em._points_corr[indices_fm[index]]) for index in indices_fm]
                        [em.imview.removeItem(em._points_corr[index]) for index in indices_fm]
                        for i in range(len(correct_em)):
                            em._points_corr[indices_fm[i]] = roi_list_em[i]
                except KeyError:
                    pass
                try:
                    indices_em = list(emdict['Correlated points indices'])
                    if len(indices_em) > 0:
                        correct_fm = list(itemgetter(*indices_em)(list(points_corr_fm)))
                        pt_list_fm = [QtCore.QPointF(p[0],p[1]) for p in np.array(correct_fm)]
                        roi_list_fm = [pg.CircleROI(pt_list_fm[i], self.fm.size_ops, parent=self.fm.imview.getImageItem(), movable=True, removable=True) for i in range(len(pt_list_fm))]
                        [roi.setPen(0,255,255) for roi in roi_list_fm]
                        [roi.removeHandle(0) for roi in roi_list_fm]
                        #[self.fm.imview.removeItem(self.fm._points_corr[indices_em[index]]) for index in indices_em]
                        [self.fm.imview.removeItem(self.fm._points_corr[index]) for index in indices_em]
                        for i in range(len(correct_fm)):
                            self.fm._points_corr[indices_em[i]] = roi_list_fm[i]
                except KeyError:
                    pass

        except KeyError:
            pass
        mdict = project['MERGE']
        self.merged = mdict['Merged']
        if self.merged:
            self.load_merge = True
            self.parent.merge(mdict)

    def _load_merge(self, mdict):
        self.popup._colors_popup = mdict['Colors']

        self.popup.c1_btn_popup.setStyleSheet('background-color: {}'.format(self.popup._colors_popup[0]))
        self.popup.c2_btn_popup.setStyleSheet('background-color: {}'.format(self.popup._colors_popup[1]))
        self.popup.c3_btn_popup.setStyleSheet('background-color: {}'.format(self.popup._colors_popup[2]))
        self.popup.c4_btn_popup.setStyleSheet('background-color: {}'.format(self.popup._colors_popup[3]))

        channels = list(mdict['Channels'])
        self.popup.channel1_btn_popup.setChecked(channels[0])
        self.popup.channel2_btn_popup.setChecked(channels[1])
        self.popup.channel3_btn_popup.setChecked(channels[2])
        self.popup.channel4_btn_popup.setChecked(channels[3])
        self.popup.channel5_btn_popup.setChecked(channels[4])
        self.popup.overlay_btn_popup.setChecked(mdict['Overlay'])

        if mdict['Max projection']:
            self.popup.max_proj_btn_popup.setChecked(True)
        else:
            self.popup.slice_select_btn_popup.setValue(mdict['Slice'])
            self.popup._slice_changed_popup()

        self.popup._update_imview_popup()

        self.popup.select_btn_popup.setChecked(True)
        points  = np.array(mdict['Selected points'])
        if len(points) > 0:
            qpoint_list = [QtCore.QPointF(p[0],p[1]) for p in points]
            [self.popup._draw_correlated_points_popup(pt, 10, self.popup.imview_popup.getImageItem()) for pt in qpoint_list]

        self.popup.select_btn_popup.setChecked(False)
        print('Data Popup:', self.popup.data_popup.shape)

    def _save_project(self):
        if self.fm.ops is not None or self.em.ops is not None:
            if self.fm.select_btn.isChecked() or self.em.select_btn.isChecked():
                buttonReply = QtWidgets.QMessageBox.question(self, 'Warning', 'Selected points have not been confirmed and will be lost during saving! \r Continue?', QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
                if buttonReply == QtWidgets.QMessageBox.Yes:
                    self._do_save()
                else:
                    pass
            else:
                self._do_save()

    def _do_save(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self,
                                                             'Save project',
                                                             self._project_folder,
                                                             '*.yml')

        if file_name is not '':
            if not '.yml' in file_name:
                file_name += '.yml'
            project = {}
            if self.fm.ops is not None:
                self._save_fm(project)
            if self.em.ops is not None:
                self._save_em(project)
            if self.fib.ops is not None:
                self._save_fib(project)
            if self.fm.ops is not None or self.em.ops is not None:
                project['MERGE'] = {}
                project['MERGE']['Merged'] = self.merged
                if self.merged:
                    self._save_merge(project['MERGE'])
            with open(file_name, 'w') as fptr:
                yaml.dump(project, fptr)

    def _save_fm(self, project):
        fmdict = {}
        project['FM'] = fmdict

        fmdict['Colors'] = self.fm._colors
        fmdict['Channels'] = self.fm._channels
        fmdict['Overlay'] = self.fm._overlay
        fmdict['Directory'] = self.fm._curr_folder
        fmdict['File'] = self.fm._file_name
        fmdict['Slice'] = self.fm._current_slice
        if self.fm._series is not None:
            fmdict['Series'] = self.fm._series
        fmdict['Align colors'] = self.fm.align_btn.isChecked()
        fmdict['Show peaks'] = self.fm.peak_btn.isChecked()
        fmdict['Show z map'] = self.fm.map_btn.isChecked()
        fmdict['Remove tilt'] = self.fm.remove_tilt_btn.isChecked()

        if self.fm.ops._show_max_proj:
            if self.fm.show_btn.isChecked():
                fmdict['Max projection orig'] = True
                fmdict['Max projection transformed'] = False
            else:
                fmdict['Max projection orig'] = False
                fmdict['Max projection transformed'] = True
        else:
            fmdict['Max projection orig'] = False
            fmdict['Max projection transformed'] = False

        fmdict['Show grid box'] = self.fm.show_grid_btn.isChecked()
        fmdict['Rotation only'] = self.fm.rot_transform_btn.isChecked()
        if self.fm.ops._orig_points is not None:
            fmdict['Original grid points'] = self.fm.ops._orig_points.tolist()
        if self.fm.ops._tf_points is not None:
            fmdict['Transformed grid points'] = self.fm.ops._tf_points.tolist()
        fmdict['Show original'] = self.fm.show_btn.isChecked()
        fmdict['Flipv'] = self.fm.flipv.isChecked()
        fmdict['Fliph'] = self.fm.fliph.isChecked()
        fmdict['Transpose'] = self.fm.transpose.isChecked()
        fmdict['Rotate'] = self.fm.rotate.isChecked()
        points = [[p.pos().x(),p.pos().y()] for p in self.fm._points_corr]
        fmdict['Correlated points'] = points
        fmdict['Original correlated points'] = self.fm._orig_points_corr
        fmdict['Correlated points indices'] = self.fm._points_corr_indices

        total_refine_matrix = np.identity(3)
        for i in range(len(self.fm.ops.refine_history)):
            total_refine_matrix = self.fm.ops.refine_history[i] @ total_refine_matrix
        fmdict['Refine matrix'] = total_refine_matrix.tolist()
        fmdict['Refine shape'] = list(self.fm.ops._refine_shape) if self.fm.ops._refine_shape is not None else None
        fmdict['Refined'] = self.fm._refined

    def _save_em(self, project):
        emdict = {}
        project['EM'] = emdict

        emdict['Directory'] = self.em._curr_folder
        emdict['File'] = self.em._file_name
        emdict['Downsampling'] = self.em._downsampling
        emdict['Show grid box'] = self.em.show_grid_btn.isChecked()
        emdict['Rotation only'] = self.em.rot_transform_btn.isChecked()
        if self.em.ops._orig_points is not None:
            emdict['Original grid points'] = self.em.ops._orig_points.tolist()
        if self.em.ops._tf_points is not None:
            emdict['Transformed grid points'] = self.em.ops._tf_points.tolist()
        emdict['Show original'] = self.em.show_btn.isChecked()
        if self.em._box_coordinate is not None:
            emdict['Subregion coordinate'] = self.em._box_coordinate.tolist()
            emdict['Select subregion original'] = self.em._select_region_original

        if self.em.ops._orig_points_region is not None:
            emdict['Orginal points subregion'] = self.em.ops._orig_points_region.tolist()
        if self.em.ops._tf_points_region is not None:
            emdict['Transformed points subregion'] = self.em.ops._tf_points_region.tolist()
        emdict['Show assembled'] = self.em.show_assembled_btn.isChecked()

        points = [[p.pos().x(),p.pos().y()] for p in self.em._points_corr]
        emdict['Correlated points'] = points
        emdict['Original correlated points '] = self.em._orig_points_corr
        emdict['Correlated points indices'] = self.em._points_corr_indices

    def _save_fib(self, project):
        fibdict = {}
        project['FIB'] = fibdict
        fibdict['Tab index'] = self.parent.em_imview.currentIndex()
        fibdict['Directory'] = self.fib._curr_folder
        fibdict['File'] = self.fib._file_name
        fibdict['Sigma angle'] = self.fib.sigma_btn.text()
        fibdict['Transpose'] = self.fib.transp_btn.isChecked()
        fibdict['Show grid'] = self.fib.show_grid_btn.isChecked()
        fibdict['Show peaks'] = self.fib.show_peaks_btn.isChecked()
        #if self.fib.ops.points is not None:
        #    fibdict['Grid points'] = self.fib.ops._grid_points_tmp.tolist() if self.fib.ops._grid_points_tmp is not None else None
        fibdict['Grid points'] = self.fib.ops.points.tolist()

        points = [[p.pos().x(),p.pos().y()] for p in self.fib._points_corr]
        fibdict['Correlated points'] = points
        fibdict['Original correlated points {}'] = self.fib._orig_points_corr
        fibdict['Correlated points indices'] = self.fib._points_corr_indices

        fibdict['Refined'] = self.fib._refined
        if self.fib._refined:
            fibdict['Refine matrix'] = self.fib.ops._refine_matrix.tolist()

    def _save_merge(self, mdict):
        mdict['Colors'] = [str(c) for c in self.popup._colors_popup]
        mdict['Channels'] = self.popup._channels_popup
        mdict['Overlay'] = self.popup._overlay_popup
        mdict['Slice'] = self.popup._current_slice_popup
        mdict['Max projection'] = self.popup.max_proj_btn_popup.isChecked()
        points = [[p.pos().x(),p.pos().y()] for p in self.popup._clicked_points_popup]
        mdict['Selected points'] = points


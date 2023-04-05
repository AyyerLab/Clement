import os

import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
from operator import itemgetter
import yaml
import copy


class Project(QtWidgets.QWidget):
    def __init__(self, fm, sem, fib, gis, tem, parent, printer, logger):
        super(Project, self).__init__()
        self._project_folder = os.getcwd()
        self.fm = fm
        self.sem = sem
        self.fib = fib
        self.gis = gis
        self.tem = tem
        self.show_fib = False
        self.merged = [False, False, False, False]
        self.popup = None
        self.parent = parent
        self.load_merge = False
        self.print = printer
        self.logger = logger
        self.tab_index = 0
        self.confirm_orientation = False

    def _load_project(self, file_name=None):
        if file_name is None:
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                 'Select project',
                                                                 self._project_folder,
                                                                 '*.yml')
        if file_name is not '':
            self.print('Load ', file_name)
            self.fm.reset_base()
            self.fm.reset_init()
            self.sem.reset_init()
            self.fib.reset_init()
            self.gis.reset_init()
            self.tem.reset_init()
            self.parent.tabs.setCurrentIndex(0)
            self._project_folder = os.path.dirname(file_name)
            with open(file_name, 'r') as f:
                project = yaml.load(f, Loader=yaml.FullLoader)
                # For numpy array debugging only!
                #project = yaml.load(f, Loader=yaml.UnsafeLoader)
            self._load_fm(project)
            self._load_em(project, sem=True)
            self._load_fib(project)
            self._load_gis(project)
            self._load_em(project, sem=False)
            #self._load_base(project)
            self.parent.tabs.setCurrentIndex(self.tab_index)

    def _load_fm(self, project):
        if 'FM' not in project:
            return
        fmdict = project['FM']
        self.fm._curr_folder = fmdict['Directory']
        self.fm._file_name = fmdict['File']
        self.fm._colors = fmdict['Colors']
        self.parent.colors = self.fm._colors
        if 'Series' in fmdict:
            self.fm._series = fmdict['Series']
        self.fm._parse_fm_images(self.fm._file_name, self.fm._series)

        undo_max_proj = False
        if fmdict['Max projection']:
            self.fm.max_proj_btn.setChecked(True)
        else:
            undo_max_proj = True

        if fmdict['Adjusted peak params']:
            self.fm.set_params_btn.click()
            self.fm.peak_controls.peak_channel_btn.setCurrentIndex(fmdict['Peak reference'])
            if fmdict['ROI']:
                self.fm.peak_controls.draw_btn.setChecked(True)
                self.fm.peak_controls.roi.setPos(np.array(fmdict['ROI pos']), update=False)
                self.fm.peak_controls.roi.setSize(np.array(fmdict['ROI shape']), update=False)
                self.fm.peak_controls.roi.setAngle(np.array(fmdict['ROI angle']), update=True)
                self.fm.peak_controls.draw_btn.setChecked(False)

            self.fm.peak_controls.t_low_label.setValue(fmdict['Low threshold'])
            self.fm.peak_controls.t_high_label.setValue(fmdict['High threshold'])
            self.fm.peak_controls.plt_label.setValue(fmdict['Min pixels threshold'])
            self.fm.peak_controls.put_label.setValue(fmdict['Max pixels threshold'])
            self.fm.peak_controls.flood_steps_label.setValue(fmdict['Flood fill steps'])
            self.fm.ops._aligned_channels = fmdict['Aligned channels']
            if len(self.fm.ops._aligned_channels) > 0:
                for i in range(self.fm.ops.num_channels):
                    if self.fm.ops._aligned_channels[i]:
                        self.fm.peak_controls.action_btns[i].setChecked(True)
            self.fm.peak_controls.save_btn.click()
            self.fm.peak_controls._update_data()
            self.fm.peak_controls.peak_btn.setChecked(True)

        try:
            pois_orig = fmdict['Pois orig']
            qpoints = [QtCore.QPointF(p[0], p[1]) for p in np.array(pois_orig)]
            #self.fm._pois_channel_indices = fmdict['Poi channel indices']

            poi_slices = fmdict['Poi slices']
            poi_channels = fmdict['Poi channel indices']
            for i in range(len(pois_orig)):
                if poi_slices[i] is not None:
                    self.fm.max_proj_btn.setEnabled(True)
                    self.fm.max_proj_btn.setChecked(False)
                    self.fm.slice_select_btn.setValue(int(poi_slices[i]))
                    self.fm._slice_changed()
                else:
                    self.fm.max_proj_btn.setChecked(True)
                if poi_channels[i] != self.fm.poi_ref_btn.currentIndex():
                    self.fm.poi_ref_btn.setCurrentIndex(poi_channels[i])
                self.fm.poi_btn.setChecked(True)
                if poi_channels[i] != self.fm.poi_ref_btn.currentIndex():
                    self.fm.poi_ref_btn.setCurrentIndex(poi_channels[i])
                self.fm._calc_pois(qpoints[i])
                self.fm.poi_btn.setChecked(False)

        except KeyError:
            pass
        try:
            if 'Original grid points' in fmdict:
                self.fm.ops._orig_points = np.array(fmdict['Original grid points'])
                self.fm.ops.points = np.copy(self.fm.ops._orig_points)
                self.fm.show_grid_btn.setEnabled(True)
                self.fm._recalc_grid()
            #self.fm.rot_transform_btn.setChecked(fmdict['Rotation only'])

            try:
                self.fm.ops._tf_points = np.array(fmdict['Transformed grid points'])
                self.fm._affine_transform(toggle_orig=False)

                if fmdict['Transpose']:
                    self.fm.transpose.setChecked(True)
                if fmdict['Rotate']:
                    self.fm.rotate.setChecked(True)
                if fmdict['Fliph']:
                    self.fm.fliph.setChecked(True)
                if fmdict['Flipv']:
                    self.fm.flipv.setChecked(True)
                if fmdict['Confirm orientation']:
                    self.confirm_orientation = True

            except KeyError:
                pass
        except KeyError:
            pass

        if 'Show peaks' in fmdict:
            if not fmdict['Show peaks']:
                self.fm.peak_btn.setChecked(False)
        self.fm.show_btn.setChecked(fmdict['Show original'])
        self.fm.show_grid_btn.setChecked(fmdict['Show grid'])
        #self.fm.map_btn.setChecked(fmdict['Show z map'])
        #self.fm.remove_tilt_btn.setChecked(fmdict['Remove tilt'])

        if undo_max_proj:
            self.fm.slice_select_btn.setValue(fmdict['Slice'])
            self.fm._slice_changed()

        self.fm._update_imview()
        self.tab_index = fmdict['Tab index']

    def _load_em(self, project, sem):
        if sem:
            if 'SEM' not in project:
                return
            emdict = project['SEM']
            em = self.sem
            tab_index = 0
        else:
            if 'TEM' not in project:
                return
            emdict = project['TEM']
            em = self.tem
            tab_index = 3

        self.parent.tabs.setCurrentIndex(tab_index)

        em._curr_folder = emdict['Directory']
        em._file_name = emdict['File']
        em.mrc_fname.setText(em._file_name)
        em.assemble_btn.setEnabled(True)
        em.step_box.setEnabled(True)
        em.step_box.setText(emdict['Downsampling'])
        em._load_mrc(jump=True)
        em._assemble_mrc()
        if emdict['Transpose']:
            em.transp_btn.setEnabled(True)
            em.transp_btn.setChecked(True)
            em._transpose()

        try:
            em._select_region_original = emdict['Select subregion original']
            try:
                em.ops._orig_points = np.array(emdict['Original grid points'])
                em.ops.points = np.copy(em.ops._orig_points)
                em.show_grid_btn.setEnabled(True)
                em._recalc_grid()
                em.show_grid_btn.setChecked(emdict['Show grid'])
                em.transform_btn.setEnabled(True)
                #em.rot_transform_btn.setChecked(emdict['Rotation only'])
            except KeyError:
                pass
            try:
                em.select_region_btn.setChecked(True)
                em._box_coordinate = np.array(emdict['Subregion coordinate'])
                em.select_region_btn.setChecked(False)
                em.ops._orig_points_region = np.array(emdict['Orginal points subregion'])
                em.ops.points = np.copy(em.ops._orig_points_region)
                em._recalc_grid()
                em.show_grid_btn.setEnabled(True)
                em.show_grid_btn.setChecked(emdict['Show grid'])
                #em.rot_transform_btn.setChecked(emdict['Rotation only'])
                try:
                    em.show_grid_btn.setChecked(False)
                    em.ops._tf_points_region = np.array(emdict['Transformed points subregion'])
                    em._affine_transform()
                except KeyError:
                    pass

                em.show_assembled_btn.setChecked(emdict['Show assembled'])
                if em.show_assembled_btn.isChecked():
                    try:
                        em.ops._tf_points = np.array(emdict['Transformed grid points'])
                        em._affine_transform()
                    except KeyError:
                        pass
            except KeyError:
                pass
        except KeyError:
            try:
                em.ops._orig_points = np.array(emdict['Original grid points'])
                em.ops.points = np.copy(em.ops._orig_points)
                em.show_grid_btn.setEnabled(True)
                em._recalc_grid()
                em.show_grid_btn.setChecked(emdict['Show grid'])
                em.transform_btn.setEnabled(True)
                #em.rot_transform_btn.setChecked(emdict['Rotation only'])
                try:
                    em.ops._tf_points = np.array(emdict['Transformed grid points'])
                    em._affine_transform()
                except KeyError:
                    pass
            except KeyError:
                pass
            try:
                em._box_coordinate = emdict['Subregion coordinate']
                em.select_region_btn.setChecked(True)
                em.select_region_btn.setChecked(False)
                em.ops._orig_points_region = np.array(emdict['Orginal points subregion'])
                em.ops.points = np.copy(em.ops._orig_points_region)
                em._recalc_grid()
                em.show_grid_btn.setChecked(emdict['Show grid'])
                #em.rot_transform_btn.setChecked(emdict['Rotation only'])
                try:
                    em.ops._tf_points_region = np.array(emdict['Transformed points subregion'])
                    em._affine_transform()
                except KeyError:
                    pass
            except KeyError:
                if em.select_region_btn.isChecked():
                    em.select_region_btn.setChecked(False)

        if self.confirm_orientation:
            self.fm.confirm_btn.setChecked(True)

        if emdict['Refined']:
            if sem:
                idx = 0
            else:
                idx = 3
            self._load_base(project, idx)

        em.show_assembled_btn.setChecked(emdict['Show assembled'])
        if not self.confirm_orientation:
            em.show_btn.setChecked(emdict['Show original'])

    def _load_fib(self, project):
        if 'FIB' not in project:
            return

        fibdict = project['FIB']
        self.parent.tabs.setCurrentIndex(1)
        self.fib._curr_folder = fibdict['Directory']
        self.fib._file_name = fibdict['File']
        self.fib.mrc_fname.setText(self.fib._file_name)
        self.fib._load_mrc(jump=True)
        if fibdict['Transpose']:
            self.fib.transp_btn.setEnabled(True)
            self.fib.transp_btn.setChecked(True)
            self.fib._transpose()  # Why has this function to be called expilicitely???

        self.fib.angle_btn.setText(fibdict['Milling angle'])
        self.fib.sem_ops = self.sem.ops

        if self.fib.sem_ops is not None and self.fib.sem_ops._orig_points is not None:
            self.fib.enable_buttons(True)

        try:
            self.fib.ops._orig_points = np.array(fibdict['Original points'])
            self.fib.show_grid_btn.setChecked(True)
            if 'Box shift' in fibdict:
                pos = self.fib.grid_box.pos()
                shift = fibdict['Box shift']
                new_pos = [pos.x() + shift[0], pos.y() + shift[1]]
                self.fib.grid_box.setPos(QtCore.QPointF(new_pos[0], new_pos[1]))
            self.fib.show_grid_btn.setChecked(fibdict['Show grid'])
        except KeyError:
            pass

        if self.tab_index == 1:
            self.show_fib = True

        if self.fib.ops.data is not None:
            if self.fib.sem_ops is not None and self.fib.sem_ops._orig_points is not None:
                self.fib.show_grid_btn.setChecked(fibdict['Show grid'])

        if fibdict['Refined']:
            self._load_base(project, idx=1)

    def _load_gis(self, project):
        if 'GIS' not in project:
            return

        gisdict = project['GIS']
        self.parent.tabs.setCurrentIndex(2)

        self.gis._curr_folder = gisdict['Directory']
        self.gis._file_name = gisdict['File']
        self.gis.mrc_fname.setText(self.gis._file_name)
        self.gis._load_mrc(jump=True)
        if gisdict['Transpose']:
            self.gis.transp_btn.setEnabled(True)
            self.gis.transp_btn.setChecked(True)
            self.gis._transpose()  # Why has this function to be called expilicitely???

        self.gis.sem_ops = self.sem.ops
        self.gis.fib_ops = self.fib.ops

        if self.gis.fib_ops is not None:
            self.gis.enable_buttons(overlay=True)
        if gisdict['Show grid']:
            self.gis.show_grid_btn.setChecked(True)
        if gisdict['Show FM peaks']:
            self.gis.show_peaks_btn.setChecked(True)
        self.gis.ops.gis_transf = np.array(gisdict['GIS transform'])
        if gisdict['Corrected']:
            self.gis.roi_pos = np.array(gisdict['ROI pos'])
            self.gis.roi_size = np.array(gisdict['ROI size'])
            self.gis.roi = pg.ROI(pos=self.gis.roi_pos, size=self.gis.roi_size, maxBounds=None, resizable=True, rotatable=False, removable=False)
            self.gis.roi.addScaleHandle(pos=[1, 1], center=[0.5, 0.5])
            self.gis.roi_pre = self.gis.fib_ops.data[self.gis.roi_pos[0]:self.gis.roi_pos[0]+self.gis.roi_size[0], self.gis.roi_pos[1]:self.gis.roi_pos[1]+self.gis.roi_size[1]]
            self.gis.roi_post = self.gis.ops.data[self.gis.roi_pos[0]:self.gis.roi_pos[0]+self.gis.roi_size[0], self.gis.roi_pos[1]:self.gis.roi_pos[1]+self.gis.roi_size[1]]
            self.gis.correct_btn.setChecked(True)

        self._load_base(project, 2)


    def _load_base(self, project, idx):
        fmdict = project['FM']
        fib_vs_sem_history = copy.copy(fmdict['FIB vs SEM history'])
        counter = [0, 0, 0]
        if idx == 0:
            emdict = project['SEM']
            em = self.sem
            em.fib = False
        elif idx == 1:
            emdict = project['FIB']
            em = self.fib
            em.fib = True
        elif idx == 3:
            emdict = project['TEM']
            em = self.tem
            em.fib = False
        for i in range(len(copy.copy(fmdict['FIB vs SEM history']))):
            if fib_vs_sem_history[i] == idx:
                self.parent.tabs.setCurrentIndex(idx)
                self.fm.other = em

                self.fm.select_btn.setChecked(True)
                em.size = emdict['Size history'][counter[idx]]
                fm_qpoints = [QtCore.QPointF(p[0], p[1]) for p in fmdict['Correlated points history'][i]]
                fm_circles = [pg.CircleROI(fm_qpoints[i], self.fm.size, parent=self.fm.imview.getImageItem(),
                                           movable=True, removable=True) for i in range(len(fm_qpoints))]
                [circle.setPen(0, 255, 0) for circle in fm_circles]
                [circle.removeHandle(0) for circle in fm_circles]
                self.fm._points_corr = copy.copy(fm_circles)
                self.fm.points_corr_z = copy.copy(fmdict['Correlated points z history'][i])
                self.fm._orig_points_corr = copy.copy(fmdict['Original correlated points history'][i])

                em_qpoints = [QtCore.QPointF(p[0], p[1]) for p in emdict['Correlated points history'][counter[idx]]]
                em_circles = [pg.CircleROI(em_qpoints[i], em.size, parent=em.imview.getImageItem(),
                                           movable=True, removable=True) for i in range(len(em_qpoints))]
                [circle.setPen(0, 255, 255) for circle in em_circles]
                [circle.removeHandle(0) for circle in em_circles]

                em._points_corr = copy.copy(em_circles)
                em.points_corr_z = copy.copy(emdict['Correlated points z history'][counter[idx]])
                em._orig_points_corr = copy.copy(emdict['Original correlated points history'][counter[idx]])
                em.ops._refine_matrix = np.array(emdict['Refinement history'][counter[idx]])

                # This is just to avoid index error when removing points during refinement
                [self.fm.anno_list.append(pg.TextItem(str(0), color=(0, 255, 0), anchor=(0, 0))) for i in
                 range(len(fm_qpoints))]
                [em.anno_list.append(pg.TextItem(str(0), color=(0, 255, 0), anchor=(0, 0))) for i in
                 range(len(fm_qpoints))]
                [self.fm._points_corr_indices.append(0) for i in range(len(fm_qpoints))]
                [em._points_corr_indices.append(0) for i in range(len(fm_qpoints))]

                self.fm.select_btn.setChecked(False)
                self.fm.counter = len(self.fm._points_corr)
                self.fm.other.counter = len(self.fm.other._points_corr)
                self.fm._refine()

                em.show_peaks_btn.setChecked(emdict['Show FM peaks'])
                counter[idx] += 1
        try:
            if self.tab_index == 1:
                emdict = project['FIB']
                em = self.fib
            else:
                if self.tab_index == 0:
                    emdict = project['SEM']
                    em = self.sem
                else:
                    emdict = project['TEM']
                    em = self.tem

            self.fm.other = em
            points_corr_fm = project['FM']['Correlated points']
            if len(points_corr_fm) != 0:
                self.fm.select_btn.setChecked(True)
                qpoints = [QtCore.QPointF(p[0], p[1]) for p in np.array(points_corr_fm)]
                [self.fm._draw_correlated_points(point, self.fm.imview.getImageItem())
                 for point in qpoints]

                #### update/correct points because in draw_correlated_points the unmoved points are drawn in other.imview
                try:
                    points_corr_em = emdict['Correlated points']
                    qpoints = [QtCore.QPointF(p[0], p[1]) for p in np.array(points_corr_em)]
                    roi_list_em = [pg.CircleROI(qpoints[i], em.size, parent=em.imview.getImageItem(),
                                                movable=True, removable=True) for i in range(len(qpoints))]
                    [roi.setPen(0, 255, 255) for roi in roi_list_em]
                    [roi.removeHandle(0) for roi in roi_list_em]

                    anno_list = [pg.TextItem(str(idx + 1), color=(0, 255, 255), anchor=(0, 0))
                                 for idx in self.fm._points_corr_indices]
                    for i in range(len(anno_list)):
                        anno_list[i].setPos(qpoints[i].x() + 5, qpoints[i].y() + 5)

                    [em.imview.removeItem(point) for point in em._points_corr]
                    [em.imview.removeItem(anno) for anno in em.anno_list]
                    [em.imview.addItem(anno) for anno in anno_list]
                    em._points_corr = roi_list_em
                    em.anno_list = anno_list
                except KeyError:
                    pass
        except KeyError:
            pass

        mdict = project['MERGE']
        self.merged = mdict['Merged']
        if self.merged[idx]:
            self.load_merge = True
            self.parent.merge(mdict)

    def _load_merge(self, mdict):
        self.popup._colors_popup = mdict['Colors']

        if mdict['Max projection']:
            self.popup.max_proj_btn_popup.setChecked(True)
        else:
            self.popup.slice_select_btn_popup.setValue(mdict['Slice'])
            self.popup._slice_changed_popup()

        self.popup._update_imview_popup()
        #if 'Points base indices' in mdict:
        #    self.popup._clicked_points_popup_base_indices = mdict['Points base indices']
        #self.popup.select_btn_popup.setChecked(True)
        #points = np.array(mdict['Selected points'])
        #if len(points) > 0:
        #    for i in range(len(points)):
        #        if i not in self.popup._clicked_points_popup_base_indices:
        #            qpoint = QtCore.QPointF(points[i][0], points[i][1])
        #            self.popup._draw_correlated_points_popup(qpoint, self.popup.imview_popup.getImageItem())

    def _save_project(self):
        if self.fm.ops is not None or self.sem.ops is not None or self.fib.ops is not None or self.gis.ops is not None or self.tem.ops is not None:
            if self.fm.select_btn.isChecked():
                buttonReply = QtWidgets.QMessageBox.question(self, 'Warning',
                                                             'Selected points have not been confirmed and will be lost during saving! \r Continue?',
                                                             QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                             QtWidgets.QMessageBox.No)
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
            if self.sem.ops is not None:
                self._save_em(project, sem=True)
            if self.fib.ops is not None:
                self._save_fib(project)
            if self.gis.ops is not None:
                self._save_gis(project)
            if self.tem.ops is not None:
                self._save_em(project, sem=False)
            project['MERGE'] = {}
            project['MERGE']['Merged'] = self.merged
            if True in self.merged:
                self._save_merge(project['MERGE'])
            self._project_folder = os.path.dirname(file_name)
            with open(file_name, 'w') as fptr:
                yaml.dump(project, fptr)

    def _save_fm(self, project):
        fmdict = {}
        project['FM'] = fmdict

        fmdict['Colors'] = np.array(self.fm._colors).tolist()
        fmdict['Channels'] = self.fm._channels
        fmdict['Overlay'] = self.fm._overlay
        fmdict['Directory'] = self.fm._curr_folder
        fmdict['File'] = self.fm._file_name
        fmdict['Slice'] = self.fm._current_slice
        if self.fm._series is not None:
            fmdict['Series'] = self.fm._series
        fmdict['Adjusted peak params'] = self.fm.ops.adjusted_params
        if self.fm.peak_controls is not None:
            fmdict['Peak reference'] = self.fm.peak_controls.peak_channel_btn.currentIndex()
            if self.fm.peak_controls.roi is not None:
                fmdict['ROI'] = True
                fmdict['ROI pos'] = self.fm.peak_controls._roi_pos.tolist()
                fmdict['ROI angle'] = self.fm.peak_controls._roi_angle
                fmdict['ROI shape'] = self.fm.peak_controls._roi_shape.tolist()
            else:
                fmdict['ROI'] = False

            fmdict['Low threshold'] = self.fm.peak_controls.t_low_label.value()
            fmdict['High threshold'] = self.fm.peak_controls.t_high_label.value()
            fmdict['Min pixels threshold'] = self.fm.peak_controls.plt_label.value()
            fmdict['Max pixels threshold'] = self.fm.peak_controls.put_label.value()
            fmdict['Flood fill steps'] = self.fm.peak_controls.flood_steps_label.value()
        fmdict['Aligned channels'] = self.fm.ops._aligned_channels
        fmdict['Show peaks'] = self.fm.peak_btn.isChecked()
        #fmdict['Show z map'] = self.fm.map_btn.isChecked()
        #fmdict['Remove tilt'] = self.fm.remove_tilt_btn.isChecked()

        fmdict['Max projection'] = self.fm.max_proj_btn.isChecked()

        fmdict['Show grid'] = self.fm.show_grid_btn.isChecked()
        #fmdict['Rotation only'] = self.fm.rot_transform_btn.isChecked()
        if self.fm.ops._orig_points is not None:
            fmdict['Original grid points'] = self.fm.ops._orig_points.tolist()
        if self.fm.ops._tf_points is not None:
            fmdict['Transformed grid points'] = self.fm.ops._tf_points.tolist()
        fmdict['Show original'] = self.fm.show_btn.isChecked()
        fmdict['Flipv'] = self.fm.flipv.isChecked()
        fmdict['Fliph'] = self.fm.fliph.isChecked()
        fmdict['Transpose'] = self.fm.transpose.isChecked()
        fmdict['Rotate'] = self.fm.rotate.isChecked()
        fmdict['Confirm orientation'] = self.fm.confirm_btn.isChecked()

        points = [[p.pos().x(), p.pos().y()] for p in self.fm._points_corr]
        fmdict['Correlated points'] = points
        fmdict['Original correlated points'] = np.array(self.fm._orig_points_corr).tolist()
        fmdict['Correlated points indices'] = self.fm._points_corr_indices
        fmdict['Correlated points history'] = [[[p.pos().x(), p.pos().y()] for p in plist] for plist in
                                               self.fm._points_corr_history]

        fmdict['Correlated points z history'] = [np.array(zhist).tolist() for zhist in self.fm._points_corr_z_history]
        fmdict['Original correlated points history'] = np.array(self.fm._orig_points_corr_history, dtype='object').tolist()
        fmdict['FIB vs SEM history'] = self.fm._fib_vs_sem_history

        fmdict['Pois orig'] = np.array(self.fm._pois_orig).tolist()
        fmdict['Poi channel indices'] = self.fm._pois_channel_indices
        fmdict['Poi slices'] = self.fm._pois_slices
        fmdict['Tab index'] = self.fm.tab_index

    def _save_em(self, project, sem):
        emdict = {}
        if sem:
            em = self.sem
            project['SEM'] = emdict
        else:
            em = self.tem
            project['TEM'] = emdict

        emdict['Tab index'] = self.parent.em_imview.currentIndex()
        emdict['Directory'] = em._curr_folder
        emdict['File'] = em._file_name
        emdict['Transpose'] = em.transp_btn.isChecked()
        emdict['Downsampling'] = em._downsampling
        emdict['Show grid'] = em.show_grid_btn.isChecked()
        #emdict['Rotation only'] = em.rot_transform_btn.isChecked()
        if em.ops._orig_points is not None:
            emdict['Original grid points'] = em.ops._orig_points.tolist()
        if em.ops._tf_points is not None:
            emdict['Transformed grid points'] = em.ops._tf_points.tolist()
        emdict['Show original'] = em.show_btn.isChecked()
        if em._box_coordinate is not None:
            emdict['Subregion coordinate'] = em._box_coordinate.tolist()
            emdict['Select subregion original'] = em._select_region_original

        if em.ops._orig_points_region is not None:
            emdict['Orginal points subregion'] = em.ops._orig_points_region.tolist()
        if em.ops._tf_points_region is not None:
            emdict['Transformed points subregion'] = em.ops._tf_points_region.tolist()
        emdict['Show assembled'] = em.show_assembled_btn.isChecked()
        emdict['Show FM peaks'] = em.show_peaks_btn.isChecked()

        points = [[p.pos().x(), p.pos().y()] for p in em._points_corr]
        emdict['Correlated points'] = points
        emdict['Original correlated points'] = np.array(em._orig_points_corr).tolist()
        emdict['Correlated points indices'] = em._points_corr_indices
        emdict['Correlated points history'] = [[[p.pos().x(), p.pos().y()] for p in plist] for plist in
                                               em._points_corr_history]

        emdict['Original correlated points history'] = [np.array(orig).tolist() for orig in em._orig_points_corr_history]
        emdict['Correlated points z history'] = [np.array(zhist).tolist() for zhist in em._points_corr_z_history]
        emdict['Size history'] = np.array(em._size_history).tolist()
        emdict['Refined'] = em._refined
        emdict['Refinement history'] = np.array(em.ops._refine_history).tolist()

    def _save_fib(self, project):
        fibdict = {}
        project['FIB'] = fibdict
        fibdict['Tab index'] = self.parent.em_imview.currentIndex()
        fibdict['Directory'] = self.fib._curr_folder
        fibdict['File'] = self.fib._file_name
        fibdict['Milling angle'] = self.fib.angle_btn.text()
        fibdict['Transpose'] = self.fib.transp_btn.isChecked()
        fibdict['Show grid'] = self.fib.show_grid_btn.isChecked()
        if self.fib.ops._orig_points is not None:
            fibdict['Original points'] = self.fib.ops._orig_points.tolist()

        if self.fib.box_shift is not None:
            fibdict['Box shift'] = self.fib.box_shift.tolist()
        if self.fib.ops._total_shift is not None:
            fibdict['Total shift'] = self.fib.ops._total_shift.tolist() if self.fib.ops._total_shift is not None else [
                0.0, 0.0]

        fibdict['Show FM peaks'] = self.fib.show_peaks_btn.isChecked()

        points = [[p.pos().x(), p.pos().y()] for p in self.fib._points_corr]
        fibdict['Correlated points'] = points
        fibdict['Original correlated points'] = self.fib._orig_points_corr
        fibdict['Correlated points indices'] = self.fib._points_corr_indices
        fibdict['Correlated points history'] = [[[p.pos().x(), p.pos().y()] for p in plist] for plist in
                                                self.fib._points_corr_history]

        fibdict['Correlated points z history'] = [np.array(zhist).tolist() for zhist in self.fib._points_corr_z_history]
        fibdict['Original correlated points history'] = [np.array(orig).tolist() for orig in self.fib._orig_points_corr_history]
        fibdict['Size history'] = np.array(self.fib._size_history).tolist()
        fibdict['Refined'] = self.fib._refined
        fibdict['Refinement history'] = np.array(self.fib.ops._refine_history).tolist()

    def _save_gis(self, project):
        gisdict = {}
        project['GIS'] = gisdict
        gisdict['Tab index'] = self.parent.em_imview.currentIndex()
        gisdict['Directory'] = self.gis._curr_folder
        gisdict['File'] = self.gis._file_name
        gisdict['Transpose'] = self.gis.transp_btn.isChecked()
        gisdict['Show grid'] = self.gis.show_grid_btn.isChecked()
        gisdict['Show FM peaks'] = self.gis.show_peaks_btn.isChecked()
        gisdict['GIS transform'] = self.gis.ops.gis_transf.tolist()
        gisdict['Corrected'] = self.gis.correct_btn.isChecked()
        gisdict['ROI pos'] = np.rint(np.array([self.gis.roi.pos().x(), self.gis.roi.pos().y()])).astype(int).tolist()
        gisdict['ROI size'] = np.rint(np.array([self.gis.roi.size().x(), self.gis.roi.size().y()])).astype(int).tolist()

    def _save_merge(self, mdict):
        mdict['Colors'] = [str(c) for c in self.popup._colors_popup]
        mdict['Channels'] = self.popup._channels_popup
        mdict['Overlay'] = self.popup._overlay_popup
        mdict['Slice'] = self.popup._current_slice_popup
        mdict['Max projection'] = self.popup.max_proj_btn_popup.isChecked()
        points = [[p.pos().x(), p.pos().y()] for p in self.popup._clicked_points_popup]
        mdict['Selected points'] = points
        mdict['Points base indices'] = self.popup._clicked_points_popup_base_indices

import os
import numpy as np
import sys
from PyQt5 import QtWidgets, QtGui, QtCore

import pyqtgraph as pg
import h5py as h5
from operator import itemgetter
from .fm_controls import FMControls
from .base_controls import BaseControls

class Project(QtWidgets.QWidget):
    def __init__(self, fm, em, parent):
        super(Project,self).__init__()
        self._project_folder = None
        self.fm = fm
        self.em = em
        self.merged = False
        self.popup = None
        self.parent = parent
        self.load_merge = False

    def _load_project(self):
        self._project_folder = os.getcwd()

        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                             'Select project',
                                                             self._project_folder,
                                                             '*.h5')
        if file_name is not '':
            self.fm.reset_init()
            self.em.reset_init()
            self._project_folder = os.path.dirname(file_name)
            with h5.File(file_name,'r') as project:
                self._load_fm(project)
                self._load_em(project)           
                self._load_base(project)                 

    def _load_fm(self, project):
        try:
            fm = project['FM']
            self.fm._curr_folder = fm.attrs['Directory']
            self.fm._file_name = fm.attrs['File']
            self.fm._colors = [n.decode() for n in list(fm['Colors'])]

            self.fm.c1_btn.setStyleSheet('background-color: {}'.format(self.fm._colors[0]))
            self.fm.c2_btn.setStyleSheet('background-color: {}'.format(self.fm._colors[1])) 
            self.fm.c3_btn.setStyleSheet('background-color: {}'.format(self.fm._colors[2])) 
            self.fm.c4_btn.setStyleSheet('background-color: {}'.format(self.fm._colors[3]))
            self.fm._channels = list(fm['Channels'])

            self.fm.channel1_btn.setChecked(self.fm._channels[0])
            self.fm.channel2_btn.setChecked(self.fm._channels[1])
            self.fm.channel3_btn.setChecked(self.fm._channels[2])
            self.fm.channel4_btn.setChecked(self.fm._channels[3])
            self.fm.overlay_btn.setChecked(fm.attrs['Overlay'])
            
            try:
                self.fm._series = fm.attrs['Series']
            except KeyError:
                pass
            self.fm._parse_fm_images(self.fm._file_name, self.fm._series)

            if fm.attrs['Max projection orig']:
                self.fm.max_proj_btn.setChecked(True)
            else:
                self.fm.slice_select_btn.setValue(fm.attrs['Slice'])
                self.fm._slice_changed()
            try:
                self.fm.align_btn.setChecked(fm.attrs['Align colors'])
            except KeyError:
                pass
            try:
                self.fm.ops._orig_points = np.array(fm['Original grid points'])
                self.fm.ops.points = np.copy(self.fm.ops._orig_points)
                self.fm.show_grid_btn.setEnabled(True)
                self.fm._recalc_grid()
                self.fm.show_grid_btn.setChecked(fm.attrs['Show grid box'])
                self.fm.rot_transform_btn.setChecked(fm.attrs['Rotation only'])
                try:
                    self.fm.ops._tf_points = np.array(fm['Transformed grid points'])
                    self.fm._affine_transform(toggle_orig=False)
                    if fm.attrs['Max projection transformed']:
                        self.fm.max_proj_btn.setChecked(True)
                    else:
                        self.fm.slice_select_btn.setValue(fm.attrs['Slice'])
                        self.fm._slice_changed()
                    self.fm.flipv.setChecked(fm.attrs['Flipv'])
                    self.fm.fliph.setChecked(fm.attrs['Fliph'])
                    self.fm.transpose.setChecked(fm.attrs['Transpose'])
                    self.fm.rotate.setChecked(fm.attrs['Rotate'])
                except KeyError:
                    pass
            except KeyError:
                pass
            try:
                self.fm.peak_btn.setChecked(fm.attrs['Show peaks'])
            except KeyError:
                pass
            self.fm.show_btn.setChecked(fm.attrs['Show original'])
            self.fm.map_btn.setChecked(fm.attrs['Show z map'])
        except KeyError:
            pass

    def _load_em(self, project):
        try:
            em = project['EM']
            self.em._curr_folder = em.attrs['Directory']
            self.em._file_name = em.attrs['File']
            self.em.mrc_fname.setText(self.em._file_name)
            self.em.assemble_btn.setEnabled(True)
            self.em.step_box.setEnabled(True)
            self.em.step_box.setText(em.attrs['Downsampling'])
            self.em._assemble_mrc()
            try:
                self.em._select_region_original = em.attrs['Select subregion original']
                try:
                    self.em.ops._orig_points = np.array(em['Original grid points'])
                    self.em.ops.points = np.copy(self.em.ops._orig_points)
                    self.em.show_grid_btn.setEnabled(True)
                    self.em._recalc_grid()
                    self.em.show_grid_btn.setChecked(em.attrs['Show grid box'])
                    self.em.rot_transform_btn.setChecked(em.attrs['Rotation only'])
                except KeyError:
                    pass
                try:
                    self.em.select_region_btn.setChecked(True)
                    self.em._box_coordinate = em.attrs['Subregion coordinate']
                    self.em.select_region_btn.setChecked(False)
                    self.em.ops._orig_points_region = np.array(em['Orginal points subregion'])
                    self.em.ops.points = np.copy(self.em.ops._orig_points_region)
                    self.em._recalc_grid()
                    self.em.show_grid_btn.setEnabled(True)
                    self.em.show_grid_btn.setChecked(em.attrs['Show grid box'])
                    self.em.rot_transform_btn.setChecked(em.attrs['Rotation only'])
                    try:
                        self.em.show_grid_btn.setChecked(False)
                        self.em.ops._tf_points_region = np.array(em['Transformed points subregion'])
                        self.em._affine_transform()
                    except KeyError:
                        pass
                     
                    self.em.show_assembled_btn.setChecked(em.attrs['Show assembled'])
                    if self.em.show_assembled_btn.isChecked():
                        try:
                            self.em.ops._tf_points = np.array(em['Transformed grid points'])
                            self.em._affine_transform()
                        except KeyError:
                            pass                       
                except KeyError:
                    pass
            except KeyError:
                try:
                    self.em.ops._orig_points = np.array(em['Original grid points'])
                    self.em.ops.points = np.copy(self.em.ops._orig_points)
                    self.em.show_grid_btn.setEnabled(True)
                    self.em._recalc_grid()
                    self.em.show_grid_btn.setChecked(em.attrs['Show grid box'])
                    self.em.rot_transform_btn.setChecked(em.attrs['Rotation only'])
                    try:
                        self.em.ops._tf_points = np.array(em['Transformed grid points'])
                        self.em._affine_transform()
                    except KeyError:
                        pass
                except KeyError:
                    pass     
                try:
                    self.em._box_coordinate = em.attrs['Subregion coordinate']
                    self.em.select_region_btn.setChecked(True)
                    self.em.select_region_btn.setChecked(False)
                    self.em.ops._orig_points_region = np.array(em['Orginal points subregion'])
                    self.em.ops.points = np.copy(self.em.ops._orig_points_region)
                    self.em._recalc_grid()
                    self.em.show_grid_btn.setChecked(em.attrs['Show grid box'])
                    self.em.rot_transform_btn.setChecked(em.attrs['Rotation only'])
                    try:
                        self.em.ops._tf_points_region = np.array(em['Transformed points subregion'])
                        self.em._affine_transform()
                    except KeyError:
                        pass
                except KeyError:
                    if self.em.select_region_btn.isChecked():
                        self.em.select_region_btn.setChecked(False)
            
            self.em.show_assembled_btn.setChecked(em.attrs['Show assembled'])
            self.em.show_btn.setChecked(em.attrs['Show original'])
        except KeyError:
            pass

    def _load_base(self, project):
        try:
            fm = project['FM']
            em = project['EM']
            num_refinements = fm.attrs['Number of refinements']
            if num_refinements == 0:
                num_refinements = 1 #make for loop go to draw points without refinement
            for i in range(num_refinements):
                try:
                    self.fm.select_btn.setChecked(True)
                    points_corr_fm = fm['Correlated points {}'.format(i)]
                    indices_fm = list(fm['Correlated points indices {}'.format(i)])
                    if len(indices_fm) > 0:
                        points_red = list(itemgetter(*indices_fm)(list(points_corr_fm)))
                        roi_list = [QtCore.QPointF(p[0],p[1]) for p in np.array(points_red)]
                        [self.fm._draw_correlated_points(roi, self.fm.size_ops, self.fm.size_other, self.fm.imview.getImageItem()) for roi in roi_list]
                except KeyError:
                    pass
                try:
                    self.em.select_btn.setChecked(True)
                    points_corr_em = em['Correlated points {}'.format(i)]
                    indices_em = list(em['Correlated points indices {}'.format(i)])
                    if len(indices_em) > 0:
                        points_red = list(itemgetter(*indices_em)(list(points_corr_em)))
                        roi_list = [QtCore.QPointF(p[0],p[1]) for p in np.array(points_red)]
                        [self.em._draw_correlated_points(roi, self.em.size_ops, self.em.size_other, self.em.imview.getImageItem()) for roi in roi_list]
                except KeyError:
                    pass
                self.em.select_btn.setChecked(False)
                self.fm.select_btn.setChecked(False)
                #### update/correct points because in draw_correlated_points the unmoved points are drawn in other.imview
                try: #do this only when correlated points exist
                    indices_fm = list(fm['Correlated points indices {}'.format(i)])

                    if len(indices_fm) > 0:
                        correct_em = list(itemgetter(*indices_fm)(list(points_corr_em)))
                        pt_list_em = [QtCore.QPointF(p[0],p[1]) for p in np.array(correct_em)]
                        roi_list_em = [pg.CircleROI(pt_list_em[i],self.em.size_ops, parent=self.em.imview.getImageItem(), movable=True, removable=True) for i in range(len(pt_list_em))]
                        [self.em.imview.removeItem(self.em._points_corr[indices_fm[index]]) for index in indices_fm]
                        for i in range(len(correct_em)):
                            self.em._points_corr[indices_fm[i]] = roi_list_em[i]
                except KeyError:
                    pass
                try:
                    indices_em = list(em['Correlated points indices {}'.format(i)])
                    if len(indices_em) > 0:
                        correct_fm = list(itemgetter(*indices_em)(list(points_corr_fm)))
                        pt_list_fm = [QtCore.QPointF(p[0],p[1]) for p in np.array(correct_fm)]
                        roi_list_fm = [pg.CircleROI(pt_list_fm[i], self.fm.size_ops, parent=self.fm.imview.getImageItem(), movable=True, removable=True) for i in range(len(pt_list_fm))]
                        [self.fm.imview.removeItem(self.fm._points_corr[indices_em[index]]) for index in indices_em]
                        for i in range(len(correct_fm)):
                            self.fm._points_corr[indices_em[i]] = roi_list_fm[i]
                except KeyError:
                    pass
                if fm.attrs['Refined']:    
                    self.fm._refine()
        except KeyError:
            pass    
        merge = project['MERGE']
        self.merged = merge.attrs['Merged']
        if self.merged:
            self.load_merge = True
            self.parent.merge(project=merge)

    def _load_merge(self, merge):
        self.popup.close()
        self.popup._colors_popup = [n.decode() for n in list(merge['Colors'])]

        self.popup.c1_btn_popup.setStyleSheet('background-color: {}'.format(self.popup._colors_popup[0]))
        self.popup.c2_btn_popup.setStyleSheet('background-color: {}'.format(self.popup._colors_popup[1])) 
        self.popup.c3_btn_popup.setStyleSheet('background-color: {}'.format(self.popup._colors_popup[2])) 
        self.popup.c4_btn_popup.setStyleSheet('background-color: {}'.format(self.popup._colors_popup[3]))
       
        channels = list(merge['Channels'])
        self.popup.channel1_btn_popup.setChecked(channels[0])
        self.popup.channel2_btn_popup.setChecked(channels[1])
        self.popup.channel3_btn_popup.setChecked(channels[2])
        self.popup.channel4_btn_popup.setChecked(channels[3])
        self.popup.channel5_btn_popup.setChecked(channels[4])
        self.popup.overlay_btn_popup.setChecked(merge.attrs['Overlay'])

        if merge.attrs['Max projection']:
            self.popup.max_proj_btn_popup.setChecked(True)
        else:
            self.popup.slice_select_btn_popup.setValue(merge.attrs['Slice'])
            self.popup._slice_changed_popup()
   
        self.popup.select_btn_popup.setChecked(True)
        points  = np.array(merge['Selected points'])
        if len(points) > 0:
            qpoint_list = [QtCore.QPointF(p[0],p[1]) for p in points]
            [self.popup._draw_correlated_points_popup(pt, 10, self.popup.imview_popup.getImageItem()) for pt in qpoint_list]

        self.popup.select_btn_popup.setChecked(False)
        self.popup._update_imview_popup()

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
                                                              '*.h5')
 
        if file_name is not '':
            file_name = os.path.splitext(file_name)[0]
            print(file_name)
            self._project_folder = os.path.dirname(file_name)
            with h5.File(file_name+'.h5','w') as project:
                if self.fm.ops is not None:
                    self._save_fm(project)     
                if self.em.ops is not None:
                    self._save_em(project)
                if self.fm.ops is not None or self.em.ops is not None:
                    merged = project.create_group('MERGE')
                    merged.attrs['Merged'] = self.merged
                    if self.merged:
                        self._save_merge(merged)
                    
                    
    def _save_fm(self, project):
        fm = project.create_group('FM')
        fm.create_dataset('Colors', data = [n.encode('ascii','ignore') for n in self.fm._colors])
        fm.create_dataset('Channels', data = self.fm._channels)
        fm.attrs['Overlay'] = self.fm._overlay
        fm.attrs['Directory'] = self.fm._curr_folder
        fm.attrs['File'] = self.fm._file_name
        fm.attrs['Slice'] = self.fm._current_slice
        if self.fm._series is not None:
            fm.attrs['Series'] = self.fm._series
        fm.attrs['Align colors'] = self.fm.align_btn.isChecked()
        fm.attrs['Show peaks'] = self.fm.peak_btn.isChecked()
        fm.attrs['Show z map'] = self.fm.map_btn.isChecked()
        
        if self.fm.ops._show_max_proj: 
            if self.fm.show_btn.isChecked():
                fm.attrs['Max projection orig'] = True
                fm.attrs['Max projection transformed'] = False
            else:
                fm.attrs['Max projection orig'] = False
                fm.attrs['Max projection transformed'] = True
        else:
            fm.attrs['Max projection orig'] = False
            fm.attrs['Max projection transformed'] = False

        fm.attrs['Show grid box'] = self.fm.show_grid_btn.isChecked()
        fm.attrs['Rotation only'] = self.fm.rot_transform_btn.isChecked()
        if self.fm.ops._orig_points is not None:
            fm.create_dataset('Original grid points', data=self.fm.ops._orig_points)
        if self.fm.ops._tf_points is not None:
            fm.create_dataset('Transformed grid points', data=self.fm.ops._tf_points)
        fm.attrs['Show original'] = self.fm.show_btn.isChecked()
        fm.attrs['Flipv'] = self.fm.flipv.isChecked()
        fm.attrs['Fliph'] = self.fm.fliph.isChecked()
        fm.attrs['Transpose'] = self.fm.transpose.isChecked()
        fm.attrs['Rotate'] = self.fm.rotate.isChecked()
        if len(self.fm._refine_history) == 0 and len(self.fm._points_corr) > 0:
            self.fm._refine_history.append([self.fm._points_corr, self.fm._points_corr_indices])
        if len(self.fm._refine_history) > 0:
            for i in range(len(self.fm._refine_history)):
                points = [[p.pos().x(),p.pos().y()] for p in self.fm._refine_history[i][0]]
                corr_points = fm.create_dataset('Correlated points {}'.format(i), data=np.array(points))
                #corr_points.attrs['Circle size FM'] = self.fm._size_ops
                #corr_points.attrs['Circle size EM'] = self.fm._size_other
                fm.create_dataset('Correlated points indices {}'.format(i), data = np.array(self.fm._refine_history[i][1]))
        fm.attrs['Number of refinements'] = self.fm._refine_counter
        if self.fm._refined:
            fm.attrs['Refined'] = self.fm._refined

    def _save_em(self, project):     
        em = project.create_group('EM')
        em.attrs['Directory'] = self.em._curr_folder
        em.attrs['File'] = self.em._file_name
        em.attrs['Downsampling'] = self.em._downsampling
        em.attrs['Show grid box'] = self.em.show_grid_btn.isChecked()
        em.attrs['Rotation only'] = self.em.rot_transform_btn.isChecked()
        if self.em.ops._orig_points is not None:
            em.create_dataset('Original grid points', data=self.em.ops._orig_points)
        if self.em.ops._tf_points is not None:
            em.create_dataset('Transformed grid points', data=self.em.ops._tf_points)
        em.attrs['Show original'] = self.em.show_btn.isChecked()
        if self.em._box_coordinate is not None:
            em.attrs['Subregion coordinate'] = self.em._box_coordinate
            em.attrs['Select subregion original'] = self.em._select_region_original

        if self.em.ops._orig_points_region is not None:
            em.create_dataset('Orginal points subregion', data=self.em.ops._orig_points_region)
        if self.em.ops._tf_points_region is not None:
            em.create_dataset('Transformed points subregion', data=self.em.ops._tf_points_region)
        em.attrs['Show assembled'] = self.em.show_assembled_btn.isChecked() 
        if len(self.em._refine_history) == 0 and len(self.em._points_corr) > 0:
            self.em._refine_history.append([self.em._points_corr, self.em._points_corr_indices])
        if len(self.em._refine_history) > 0:
            for i in range(len(self.em._refine_history)):
                points = [[p.pos().x(),p.pos().y()] for p in self.em._refine_history[i][0]]
                points_corr = em.create_dataset('Correlated points {}'.format(i), data=np.array(np.copy(points)))
                #points_corr.attrs['Circle size EM'] = self.em._size_ops
                #points_corr.attrs['Circle size FM'] = self.em._size_other
                em.create_dataset('Correlated points indices {}'.format(i), data = np.array(self.em._refine_history[i][1]))
             
    def _save_merge(self,merged):
        merged.create_dataset('Colors', data = [n.encode('ascii','ignore') for n in self.popup._colors_popup])
        merged.create_dataset('Channels', data = self.popup._channels_popup)
        print(self.popup._overlay_popup)
        print(self.popup._channels_popup)
        merged.attrs['Overlay'] = self.popup._overlay_popup
        merged.attrs['Slice'] = self.popup._current_slice_popup
        merged.attrs['Max projection'] = self.popup.max_proj_btn_popup.isChecked()
        points = [[p.pos().x(),p.pos().y()] for p in self.popup._clicked_points_popup]
        merged.create_dataset('Selected points', data = np.array(points))
               

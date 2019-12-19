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
    def __init__(self, fm, em):
        super(Project,self).__init__()
        self._project_folder = None
        self.fm = fm
        self.em = em
        self.merged = False

    def _load_project(self):
        self._project_folder = os.getcwd()

        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                             'Select project',
                                                             self._project_folder,
                                                             '*.h5')
        if file_name is not '':
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
            self.fm._current_slice = fm.attrs['Slice']
            self.fm._series = fm.attrs['Series']
            self.fm._parse_fm_images(self.fm._file_name, self.fm._series)
            if fm.attrs['Max projection orig']:
                self.fm.max_proj_btn.setChecked(fm.attrs['Max projection'])
            else:
                self.fm.slice_select_btn.setValue(fm.attrs['Slice'])
                self.fm._slice_changed()
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
                        self.fm.max_proj_btn.setChecked(fm.attrs['Max projection'])
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
            self.fm.show_btn.setChecked(fm.attrs['Show original'])
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
                print(self.em._select_region_original)
                if self.em._select_region_original:
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
                else:
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
                        self.em.select_region_btn.setChecked(True)
                        self.em._box_coordinate = em.attrs['Subregion coordinate']
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
                        pass
                    self.em.show_assembled_btn.setChecked(em.attrs['Show assembled'])
            except KeyError:
                pass



            self.em.show_btn.setChecked(em.attrs['Show original'])
        except KeyError:
            pass

    def _load_base(self, project):
        try:
            fm = project['FM']
            em = project['EM']
            num_refinements = fm.attrs['Number of refinements']
            for i in range(num_refinements+1):
                print(i)
                try:
                    self.fm.select_btn.setChecked(True)
                    points_corr_fm = fm['Correlated points {}'.format(i)]
                    indices_fm = list(fm['Correlated points indices {}'.format(i)])
                    if len(indices_fm) > 0:
                        points_red = list(itemgetter(*indices_fm)(list(points_corr_fm)))
                        roi_list = [QtCore.QPointF(p[0],p[1]) for p in np.array(points_red)]
                        [self.fm._draw_correlated_points(roi, points_corr_fm.attrs['Circle size FM'], points_corr_fm.attrs['Circle size EM'], self.fm.imview.getImageItem()) for roi in roi_list]
                except KeyError:
                    pass
                try:
                    self.em.select_btn.setChecked(True)
                    points_corr_em = em['Correlated points {}'.format(i)]
                    indices_em = list(em['Correlated points indices {}'.format(i)])
                    if len(indices_em) > 0:
                        points_red = list(itemgetter(*indices_em)(list(points_corr_em)))
                        roi_list = [QtCore.QPointF(p[0],p[1]) for p in np.array(points_red)]
                        [self.em._draw_correlated_points(roi, points_corr_em.attrs['Circle size EM'], points_corr_em.attrs['Circle size FM'], self.em.imview.getImageItem()) for roi in roi_list]
                except KeyError:
                    pass
                self.em.select_btn.setChecked(False)
                self.fm.select_btn.setChecked(False)
                if fm.attrs['Refined']:
                    #### update/correct points
                    if len(indices_fm) > 0:
                        correct_em = list(itemgetter(*indices_fm)(list(points_corr_em)))
                        pt_list_em = [QtCore.QPointF(p[0],p[1]) for p in np.array(correct_em)]
                        roi_list_em = [pg.CircleROI(pt_list_em[i],points_corr_fm.attrs['Circle size EM'], parent=self.em.imview.getImageItem(), movable=True, removable=True) for i in range(len(pt_list_em))]
                        [self.em.imview.removeItem(self.em._points_corr[indices_fm[index]]) for index in indices_fm]
                        for i in range(len(correct_em)):
                            self.em._points_corr[indices_fm[i]] = roi_list_em[i]
                        [print(p.pos()) for p in self.fm._points_corr]
                    if len(indices_em) > 0:
                        correct_fm = list(itemgetter(*indices_em)(list(points_corr_fm)))
                        pt_list_fm = [QtCore.QPointF(p[0],p[1]) for p in np.array(correct_fm)]
                        roi_list_fm = [pg.CircleROI(pt_list_fm[i],points_corr_em.attrs['Circle size EM'], parent=self.fm.imview.getImageItem(), movable=True, removable=True) for i in range(len(pt_list_fm))]
                        [self.fm.imview.removeItem(self.fm._points_corr[indices_em[index]]) for index in indices_em]
                        for i in range(len(correct_fm)):
                            self.fm._points_corr[indices_em[i]] = roi_list_fm[i]
                        [print(p.pos()) for p in self.em._points_corr]
                    print(np.array(points_corr_em))
                    self.fm._refine()
        except KeyError:
            pass    
        base = project['BASE']
        self.merged = base.attrs['Merged']


    

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
            self._project_folder = os.path.dirname(file_name)
            with h5.File(file_name+'.h5','w') as project:
                if self.fm.ops is not None:
                    self._save_fm(project)     
                if self.em.ops is not None:
                    self._save_em(project)
                if self.fm.ops is not None:
                    base = project.create_group('BASE')
                    base.attrs['Merged'] = self.fm._merged
                elif self.em.ops is not None:
                    base = project.create_group('BASE')
                    base.attrs['Merged'] = self.fm._merged
                    
                    
    def _save_fm(self, project):
        fm = project.create_group('FM')
        fm.attrs['Directory'] = self.fm._curr_folder
        fm.attrs['File'] = self.fm._file_name
        fm.attrs['Slice'] = self.fm._current_slice
        fm.attrs['Series'] = self.fm._series
        if self.fm.ops._show_max_proj: 
            if self.fm.show_btn.isClicked():
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
                corr_points.attrs['Circle size FM'] = self.fm._size_ops
                corr_points.attrs['Circle size EM'] = self.fm._size_other
                fm.create_dataset('Correlated points indices {}'.format(i), data = np.array(self.fm._refine_history[i][1]))
        fm.attrs['Number of refinements'] = len(self.fm._refine_history)-1 #points_corr are saved in history before refinement --> -1
        if self.fm._refine:
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
        print(self.em._refine_history)
        print(self.fm._refine_history)
        if len(self.em._refine_history) > 0:
            for i in range(len(self.em._refine_history)):
                points = [[p.pos().x(),p.pos().y()] for p in self.em._refine_history[i][0]]
                points_corr = em.create_dataset('Correlated points {}'.format(i), data=np.array(np.copy(points)))
                points_corr.attrs['Circle size EM'] = self.em._size_ops
                points_corr.attrs['Circle size FM'] = self.em._size_other
                em.create_dataset('Correlated points indices {}'.format(i), data = np.array(self.em._refine_history[i][1]))
              

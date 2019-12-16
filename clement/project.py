import os
import numpy as np
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import h5py as h5
from .fm_controls import FMControls
from .base_controls import BaseControls

class Project(QtWidgets.QWidget):
    def __init__(self, fm, em):
        super(Project,self).__init__()
        self._project_folder = None
        self.fm = fm
        self.em = em

    def _load_project(self):
        self._project_folder = os.getcwd()

        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                             'Select project',
                                                             self._project_folder,
                                                             '*.h5')
        if file_name is not '':
            self._project_folder = os.path.dirname(file_name)
            with h5.File(file_name,'r') as project:
                #### Load FM variables 
                try:
                    fm = project['FM']
                    self.fm._curr_folder = fm.attrs['Directory']
                    self.fm._file_name = fm.attrs['File']
                    self.fm._current_slice = fm.attrs['Slice']
                    self.fm._series = fm.attrs['Series']
                    self.fm._parse_fm_images(self.fm._file_name, self.fm._series)
                    try:    
                        self.fm.ops._max_proj_data = np.array(fm['Max data'])
                    except KeyError:
                        self.fm.ops._max_proj_data = None           
                    if fm.attrs['Max projection']:
                        self.fm.max_proj_btn.setChecked(fm.attrs['Max projection'])
                    else:
                        self.fm.slice_select_btn.setValue(fm.attrs['Slice'])
                        self.fm._slice_changed()
                    
                    self.fm.ops._transformed = fm.attrs['Transformed']
                    try:
                        self.fm.ops._orig_points = np.array(fm['Original grid points'])
                        self.fm.show_grid_btn.setEnabled(True)
                        try:
                            self.fm.ops._tf_points = np.array(fm['Transformed grid points'])
                        except KeyError:
                            pass
                        if self.fm.ops._transformed:
                            self.fm.ops.points = np.copy(self.fm.ops._tf_points)
                        else:
                            self.fm.ops.points = np.copy(self.fm.ops._orig_points)
                        
                        self.fm._recalc_grid()
                        self.fm.show_grid_btn.setChecked(fm.attrs['Show grid box'])
                    except KeyError:
                        pass
                except KeyError:
                    pass
                    
                #### Load EM variables
                try:
                    em = project['EM']
                    self.em._curr_folder = em.attrs['Directory']
                    self.em._file_name = em.attrs['File']
                    self.em.mrc_fname.setText(self.em._file_name)
                    self.em.assemble_btn.setEnabled(True)
                    self.em.step_box.setEnabled(True)
                    self.em.step_box.setText(em.attrs['Downsampling'])
                    self.em._assemble_mrc()
                    self.fm.ops._transformed = em.attrs['Transformed']
                    
                    try:
                        self.em.ops._orig_points = em['Original grid points']
                        self.em.show_grid_btn.setEnabled(True)
                        try:
                            self.em.ops._tf_points = np.array(em['Transformed grid points'])
                        except KeyError:
                            pass
                        if self.em.ops._transformed:
                            self.em.ops.points = np.copy(self.fm.ops._tf_points)
                        else:
                            self.em.ops.points = np.copy(self.fm.ops._orig_points)
                        self.em._recalc_grid()
                        self.em.show_grid_btn.setChecked(em['Show grid box'])
                    except KeyError:
                        pass 
                except KeyError:
                    pass



        
    def _save_project(self):
        if self.fm.ops is not None or self.em.ops is not None:
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self,
                                                                  'Save project',
                                                                  self._project_folder,
                                                                  '*.h5')
                
            if file_name is not '':
                self._project_folder = os.path.dirname(file_name)
                with h5.File(file_name+'.h5','w') as project:
                    if self.fm.ops is not None:
                        fm = project.create_group('FM')
                        fm.attrs['Directory'] = self.fm._curr_folder
                        fm.attrs['File'] = self.fm._file_name
                        fm.attrs['Slice'] = self.fm._current_slice
                        fm.attrs['Series'] = self.fm._series
                        fm.attrs['Max projection'] = self.fm.ops._show_max_proj
                        fm.attrs['Show grid box'] = self.fm.show_grid_btn.isChecked()
                        print(self.fm.show_grid_btn.isChecked())
                        fm.attrs['Transformed'] = self.fm.ops._transformed
                        if self.fm.ops._max_proj_data is not None:
                            fm.create_dataset('Max data', data=self.fm.ops._max_proj_data)
                        if self.fm.ops._orig_points is not None:
                            fm.create_dataset('Original grid points', data=self.fm.ops._orig_points)
                        if self.fm.ops._tf_points is not None:
                            fm.create_dataset('Transformed grid points', data=self.fm.ops._tf_points)
                    if self.em.ops is not None:
                        em = project.create_group('EM')
                        em.attrs['Directory'] = self.em._curr_folder
                        em.attrs['File'] = self.em._file_name
                        em.attrs['Downsampling'] = self.em._downsampling
                        em.attrs['Transformed'] = self.em.ops._transformed
                        if self.em.ops._orig_points is not None:
                            em.create_dataset('Original grid points', data=self.em.ops._orig_points)
                        if self.em.ops._tf_points is not None:
                            em.create_dataset('Transformed grid points', data=self.em.ops._tf_points)           
        
        
        
        
        
        

import numpy as np
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import scipy.ndimage as ndi
import copy

class BaseControls(QtWidgets.QWidget):
    def __init__(self):
        super(BaseControls, self).__init__()
        # ops is EM_operations or FM_operations
        self.tag = 'base'
        self.ops = None
        self.other = None # The other controls object
        
        self._box_coordinate = None
        self._points_corr = []
        self.orig_points_corr = []
        self._points_corr_indices= []
        self._refined = False
        self._refine_history = []
        self._refine_counter = 0 
        #self._merged = False
        
        self.tr_matrices = None
        self.show_grid_box = False
        self.show_tr_grid_box = False
        self.clicked_points = []
        self.grid_box = None
        self.tr_grid_box = None
        self.boxes = []
        self.tr_boxes = []
        self.original_help = True
        self.redo_tr = False
        self.setContentsMargins(0, 0, 0, 0)
        self.counter = 0
        self.anno_list = []
        self.size_ops = 10
        self.size_other = 10

    def _init_ui(self):
        print('This message should not be seen. Please override _init_ui')

    def _imview_clicked(self, event):

        print(self)
        if event.button() == QtCore.Qt.RightButton:
            event.ignore()
            return

        if self.ops is None:
            return

        pos = self.imview.getImageItem().mapFromScene(event.pos())

        #self.size_ops = self.ops.data.shape[0]*0.01
        #if self.other.ops is not None:
            #self.size_other = self.other.ops.data.shape[0]*0.005
            #self.size_other = self.size_ops
        #   pass
        item = self.imview.getImageItem()

        pos.setX(pos.x() - self.size_ops/2)
        pos.setY(pos.y() - self.size_ops/2)
        if hasattr(self, 'define_btn') and self.define_btn.isChecked():
            roi = pg.CircleROI(pos, self.size_ops, parent=item, movable=False)
            roi.setPen(255,0,0)
            roi.removeHandle(0)
            self.imview.addItem(roi)
            self.clicked_points.append(roi)
        elif self.select_btn.isChecked():
            self._draw_correlated_points(pos, self.size_ops, self.size_other, item)
        elif hasattr(self, 'select_region_btn') and self.select_region_btn.isChecked():
            '''EM only: Select individual image from montage'''
            points_obj = (pos.x(), pos.y())
            self._box_coordinate = np.copy(np.array(points_obj))

            # If clicked point is inside image
            if points_obj[0] < self.ops.data.shape[0] and points_obj[1] < self.ops.data.shape[1]:
                # Get index of selected region
                ind = self.ops.get_selected_region(np.array(points_obj), not self.show_btn.isChecked())

                # If index is non-ambiguous
                if ind is not None:
                    if self.show_btn.isChecked():
                        boxes = self.boxes
                    else:
                        boxes = self.tr_boxes
                    white_pen = pg.mkPen('w')
                    [box.setPen(white_pen) for box in boxes]
                    if ind < len(boxes):
                        boxes[ind].setPen(pg.mkPen('#ed7370'))
                    else:
                        print('Oops, something went wrong. Try again!')
            else:
                print('Oops, something went wrong. Try again!')

    def _draw_correlated_points(self, pos, size1, size2, item):
        if self.other.ops is None:
            print('Select both data first')
        
        else:
            if self.ops._transformed and self.other.ops._transformed:
                if self.tr_matrices is not None:
                    if hasattr(self, 'peak_btn') and self.peak_btn.isChecked():
                        ind = self.ops.check_peak_index(np.array((pos.x(), pos.y())), size1)
                        if ind is not None:
                            pos.setX(self.ops.peaks_2d[ind,0] - size1 / 2)
                            pos.setY(self.ops.peaks_2d[ind,1] - size1 / 2)

                    point_obj = pg.CircleROI(pos, size1, parent=item, movable=False, removable=True)
                    point_obj.setPen(0,255,0)
                    point_obj.removeHandle(0)
                    self.imview.addItem(point_obj)
                    self._points_corr.append(point_obj)
                    self.counter += 1
                    annotation_obj = pg.TextItem(str(self.counter), color=(0,255,0), anchor=(0,0))
                    annotation_obj.setPos(pos.x()+5, pos.y()+5)
                    self.imview.addItem(annotation_obj)
                    self.anno_list.append(annotation_obj)

                    self._points_corr_indices.append(self.counter-1)

                    # Coordinates in clicked image
                    init = np.array([point_obj.x()+size1/2,point_obj.y()+size1/2, 1])

                    transf = np.dot(self.tr_matrices, init)
                    pos = QtCore.QPointF(transf[0]-size2/2, transf[1]-size2/2)
                    point_other = pg.CircleROI(pos, size2, parent=self.other.imview.getImageItem(), movable=True, removable=True)
                    point_other.setPen(0,255,255)
                    point_other.removeHandle(0)
                    self.other.imview.addItem(point_other)
                    self.other._points_corr.append(point_other)

                    self.other.counter = self.counter
                    annotation_other = pg.TextItem(str(self.counter), color=(0,255,255), anchor=(0,0))
                    annotation_other.setPos(pos.x()+5, pos.y()+5)
                    self.other.imview.addItem(annotation_other)
                    self.other.anno_list.append(annotation_other)

                    point_obj.sigRemoveRequested.connect(lambda: self._remove_correlated_points(self.imview, self.other.imview, point_obj, point_other, self._points_corr, self.other._points_corr, annotation_obj, annotation_other, self.anno_list, self.other.anno_list))
                    point_other.sigRemoveRequested.connect(lambda: self._remove_correlated_points(self.other.imview, self.imview, point_other, point_obj, self.other._points_corr, self._points_corr, annotation_other, annotation_obj, self.other.anno_list, self.anno_list))

            else:
                print('Transform both images before point selection')

    def _remove_correlated_points(self,imv1,imv2,pt1,pt2,pt_list,pt_list2, anno, anno2, anno_list, anno_list2):
        imv1.removeItem(pt1)
        imv2.removeItem(pt2)
        pt_list.remove(pt1)
        pt_list2.remove(pt2)
        imv1.removeItem(anno)
        imv2.removeItem(anno2)
        anno_list.remove(anno)
        anno_list2.remove(anno2)

    def _define_grid_toggled(self, checked):
        if self.ops is None:
            print('Select data first!')
            return

        if checked:
            print('Defining grid on %s image: Click on corners'%self.tag)
            self.show_grid_btn.setChecked(False)
        else:
            print('Done defining grid on %s image: Manually adjust fine positions'%self.tag)
            if len(self.clicked_points) == 4:
                self.show_grid_btn.setEnabled(True)
                if not self.ops._transformed:
                    self.transform_btn.setEnabled(True)
                # Initialize grid_box on original or transformed image
                if self.show_btn.isChecked():
                    self.grid_box = None
                else:
                    self.tr_grid_box = None

                # Calculate circle centers
                positions = [c.pos() for c in self.clicked_points]
                sizes = [c.size()[0] for c in self.clicked_points]
                for pos, s in zip(positions, sizes):
                    pos.setX(pos.x() + s/2)
                    pos.setY(pos.y() + s/2)
                points = np.array([(point.x(), point.y()) for point in positions])

                # If original image
                if self.show_btn.isChecked():
                    self.grid_box = pg.PolyLineROI(positions, closed=True, movable=False)
                    self.grid_box.sigRegionChangeFinished.connect(self.store_grid_box_points)
                    # If assembly is an option (EM image)
                    if hasattr(self, 'show_assembled_btn'):
                        if self.show_assembled_btn.isChecked():
                            self.ops._orig_points = points
                        else:
                            self.ops._orig_points_region = points
                    else:
                        self.ops._orig_points = points
                    self.ops.points = points
                else:
                    self.tr_grid_box = pg.PolyLineROI(positions, closed=True, movable=False)
                [self.imview.removeItem(roi) for roi in self.clicked_points]
                self.clicked_points = []
                self.show_grid_btn.setEnabled(True)
                self.show_grid_btn.setChecked(True)
                if self.show_btn.isChecked():
                    self.show_grid_box = True
                else:
                    self.show_tr_grid_box = True
            else:
                print('You have to select exactly 4 points. Try again!')
                [self.imview.removeItem(roi) for roi in self.clicked_points]
                self._show_grid(None)
                self.clicked_points = []

    def _recalc_grid(self, toggle_orig=False):
        if self.ops.points is not None:
            if self.show_btn.isChecked():
                pos =  list(self.ops.points)
                poly_line = pg.PolyLineROI(pos, closed=True, movable=False)
                if self.show_grid_btn.isChecked():
                    if not toggle_orig:
                        self.imview.removeItem(self.tr_grid_box)
                        self.imview.removeItem(self.grid_box)
                        self.show_grid_box = False
                print('Recalculating original grid...')
                self.grid_box = poly_line
            else:
                pos =  list(self.ops.points)
                poly_line = pg.PolyLineROI(pos, closed=True, movable=False)
                if self.show_grid_btn.isChecked():
                    if not toggle_orig:
                        self.imview.removeItem(self.grid_box)
                        self.imview.removeItem(self.tr_grid_box)
                        self.show_tr_grid_box = False
                    if self.redo_tr:
                        self.imview.removeItem(self.tr_grid_box)
                        self.redo_tr = False
                print('Recalculating transformed grid...')
                self.tr_grid_box = poly_line
            self._show_grid(None)

    def store_grid_box_points(self):
        points_obj = self.grid_box.getState()['points']
        points = np.array([list((point[0], point[1])) for point in points_obj])
        if self.show_btn.isChecked():
            if hasattr(self, 'show_assembled_btn'):
                if self.show_assembled_btn.isChecked():
                    self.ops._orig_points = points
                else:
                    self.ops._orig_points_region = points
            else:
                self.ops._orig_points = points
            self.ops.points = points

    def _show_grid(self, state):
        if self.show_grid_btn.isChecked():
            if self.show_btn.isChecked():
                self.show_grid_box = True
                self.imview.addItem(self.grid_box)
                if self.show_tr_grid_box:
                    self.imview.removeItem(self.tr_grid_box)
                    self.show_tr_grid_box = False
            else:
                self.show_tr_grid_box = True
                self.imview.addItem(self.tr_grid_box)
                if self.show_grid_box:
                    self.imview.removeItem(self.grid_box)
                    self.show_grid_box = False
        else:
            if self.show_btn.isChecked():
                if self.show_grid_box:
                    self.imview.removeItem(self.grid_box)
                    self.show_grid_box = False
            else:
                if self.show_tr_grid_box:
                    self.imview.removeItem(self.tr_grid_box)
                    self.show_tr_grid_box = False

    def _define_corr_toggled(self, checked):
        if self.ops is None or self.other.ops is None:
            print('Select both data first')
            return

        condition = False
        if hasattr(self.ops, 'tf_region'):
            #if self.ops.tf_region is not None or self.ops.tf_data is not None:
            #    if self.other.ops.tf_data is not None or self.other.ops.tf_max_proj_data is not None or self.other.ops.tf_hsv_map is not None or self.other.ops.tf_hsv_map_no_tilt is not None:
            if self.ops._tf_points is not None or self.ops._tf_points_region is not None:
                if self.other.ops._tf_points is not None:
                    condition = True
        else:
            #if self.ops.tf_data is not None or self.ops.tf_max_proj_data is not None or self.ops.tf_hsv_map is not None or self.ops.tf_hsv_map_no_tilt is not None:
            #    if self.other.ops.tf_region is not None or self.other.ops.tf_data is not None:
            if self.ops._tf_points is not None:
                if self.other.ops._tf_points is not None or self.other.ops._tf_points_region is not None:
                    condition = True

        if condition:
            if checked:
                print('Select points of interest on %s image'%self.tag)
                if not self.other.select_btn.isChecked():
                    print(self._points_corr_indices)
                    if len(self._points_corr_indices) != 0:
                        [self.imview.removeItem(point) for point in self._points_corr]
                        [self.other.imview.removeItem(point) for point in self.other._points_corr]
                        [self.imview.removeItem(anno) for anno in self.anno_list]
                        [self.other.imview.removeItem(anno) for anno in self.other.anno_list]
                        self._points_corr = []
                        self.other._points_corr = []
                        self._points_corr_indices = []
                        self.other._points_corr_indices = []
                        self.anno_list = []
                        self.other.anno_list = []
                        self.counter = 0
                        self.other.counter = 0
                src_sorted = np.array(sorted(self.ops.points, key=lambda k: [np.cos(30*np.pi/180)*k[0] + k[1]]))

                if hasattr(self.other, 'fib') and self.other.fib:
                    dst_sorted = np.array(
                        sorted(self.other.sem_ops.points, key=lambda k: [np.cos(30 * np.pi / 180) * k[0] + k[1]]))
                    self.tr_matrices = self.other.ops.get_fib_transform(src_sorted, dst_sorted, self.other.sem_ops.tf_matrix)
                else:
                    dst_sorted = np.array(
                        sorted(self.other.ops.points, key=lambda k: [np.cos(30 * np.pi / 180) * k[0] + k[1]]))
                    self.tr_matrices = self.ops.get_transform(src_sorted, dst_sorted)
            else:
                print('Done selecting points of interest on %s image'%self.tag)
        else:
            if checked:
                print('Select and transform both data first')

    def _affine_transform(self, toggle_orig=True):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if self.show_btn.isChecked():
            grid_box = self.grid_box
        else:
            self.redo_tr = True
            grid_box = self.tr_grid_box

        if grid_box is not None:
            if hasattr(self, 'fliph'):
                self.fliph.setChecked(False)
                self.fliph.setEnabled(True)
                self.flipv.setChecked(False)
                self.flipv.setEnabled(True)
                self.transpose.setChecked(False)
                self.transpose.setEnabled(True)
                self.rotate.setChecked(False)
                self.rotate.setEnabled(True)
                self.auto_opt_btn.setEnabled(True)

            points_obj = grid_box.getState()['points']
            points = np.array([list((point[0], point[1])) for point in points_obj])
            if self.rot_transform_btn.isChecked():
                print('Performing rotation on %s image'%self.tag)
                self.ops.calc_rot_transform(points)
            else:
                print('Performing affine transformation on %s image'%self.tag)
                self.ops.calc_affine_transform(points)

            self.original_help = False
            self.show_btn.setEnabled(True)
            self.show_btn.setChecked(False)
            self.original_help = True
            self._recalc_grid(toggle_orig=toggle_orig)
            self._update_imview()
            self.transform_btn.setEnabled(False)
            self.rot_transform_btn.setEnabled(False)
        else:
            print('Define grid box on %s image first!'%self.tag)
        QtWidgets.QApplication.restoreOverrideCursor()

    def _show_original(self, state):
        if self.original_help:
            if self.ops is not None:
                self.ops._transformed = not self.ops._transformed
                if hasattr(self.ops, 'flipv') and not self.ops._transformed:
                    self.flipv.setEnabled(False)
                    self.fliph.setEnabled(False)
                    self.transpose.setEnabled(False)
                    self.rotate.setEnabled(False)
                elif hasattr(self.ops, 'flipv') and self.ops._transformed:
                    if not self.ops.refined:
                        self.flipv.setEnabled(True)
                        self.fliph.setEnabled(True)
                        self.transpose.setEnabled(True)
                        self.rotate.setEnabled(True)

                print('Transformed?', self.ops._transformed)
                if hasattr(self.ops,'refined'):
                    if self.ops.refined:
                        self.ops.toggle_original(update=False)
                    else:
                        self.ops.toggle_original()
                else:
                    self.ops.toggle_original()
                self._recalc_grid(toggle_orig=True)

                self._update_imview()
                if self.ops._transformed:
                    self.transform_btn.setEnabled(False)
                    self.rot_transform_btn.setEnabled(False)
                else:
                    self.transform_btn.setEnabled(True)
                    self.rot_transform_btn.setEnabled(True)

    def _refine(self):

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        if len(self._points_corr) > 3:
            if not self.select_btn.isChecked() and not self.other.select_btn.isChecked():
                    
                self._refine_history.append([self._points_corr, self._points_corr_indices])
                self.other._refine_history.append([self.other._points_corr, self.other._points_corr_indices])
                self._refine_counter += 1
                print('Refining...')
                dst = np.array([[point.x()+self.size_other/2,point.y()+self.size_other/2] for point in self.other._refine_history[-1][0]])
                src = np.array([[point.x()+self.size_ops/2,point.y()+self.size_ops/2] for point in self._refine_history[-1][0]])
                
                self.ops.calc_refine_matrix(src, dst,self.other.ops.points)
                [self.imview.removeItem(point) for point in self._points_corr]
                [self.other.imview.removeItem(point) for point in self.other._points_corr]
                [self.imview.removeItem(anno) for anno in self.anno_list]
                [self.other.imview.removeItem(anno) for anno in self.other.anno_list]

                self.auto_opt_btn.setChecked(False)
                self.anno_list = []
                self.other.anno_list = []
                self.counter = 0
                self.other.counter = 0
                self._points_corr = []
                self.other._points_corr = []
                self._points_corr_indices = []
                self.other._points_corr_indices = []
                self._refined = True
                self._recalc_grid()
                self._update_imview()
                self.fliph.setEnabled(False)
                self.flipv.setEnabled(False)
                self.transpose.setEnabled(False)
                self.rotate.setEnabled(False)
            else:
                print('Confirm point selection! (Uncheck Select points of interest)')
        else:
            print('Select at least 4 points for refinement!')
        QtWidgets.QApplication.restoreOverrideCursor()

    def _optimize(self):
        if self.auto_opt_btn.isChecked():
            self.orig_points_corr = copy.copy(self._points_corr)
            self.other.orig_points_corr = copy.copy(self.other._points_corr)

            em_points = np.round(np.array([[point.x()+self.size_ops/2,point.y()+self.size_ops/2] for point in self.other._points_corr])).astype(np.int)
            fm_points = np.round(np.array([[point.x()+self.size_ops/2,point.y()+self.size_ops/2] for point in self._points_corr])).astype(np.int)
            if self.ops.data.ndim>2:
                fm_max = np.max(self.ops.data, axis=-1)
            else:
                fm_max = self.ops.data
 
            [self.imview.removeItem(point) for point in self._points_corr]
            [self.other.imview.removeItem(point) for point in self.other._points_corr]         
            self._points_corr = []
            self.other._points_corr = []
            fm_points, em_points = self.ops.optimize(fm_max, self.other.ops.data, fm_points, em_points)         
            for i in range(len(fm_points)):
                pos_fm = QtCore.QPointF(fm_points[i][0]-self.size_ops/2, fm_points[i][1]-self.size_ops/2)
                pos_em = QtCore.QPointF(em_points[i][0]-self.size_ops/2, em_points[i][1]-self.size_ops/2)
                point_fm = pg.CircleROI(pos_fm, self.size_ops, parent=self.imview.getImageItem(), movable=False, removable=True)
                point_fm.removeHandle(0)
                point_em = pg.CircleROI(pos_em, self.size_other, parent=self.imview.getImageItem(), movable=False, removable=True)
                point_em.removeHandle(0)
                self._points_corr.append(point_fm)
                self.other._points_corr.append(point_em)
                self.imview.addItem(point_fm)
                self.other.imview.addItem(point_em)
        else:
            [self.imview.removeItem(point) for point in self._points_corr]
            [self.other.imview.removeItem(point) for point in self.other._points_corr]     
            
            self._points_corr = copy.copy(self.orig_points_corr)
            self.other._points_corr = copy.copy(self.other.orig_points_corr)
            

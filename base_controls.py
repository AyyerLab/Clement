import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

class BaseControls(QtWidgets.QWidget):
    def __init__(self):
        super(BaseControls, self).__init__()
        # ops is EM_operations or FM_operations
        self.tag = 'base'
        self.ops = None
        self.other = None # The other controls object

        self.tr_matrices = None
        self.clicked_points = []
        self.points_corr = []
        self.show_grid_box = False
        self.show_tr_grid_box = False
        self.grid_box = None
        self.tr_grid_box = None
        self.boxes = []
        self.tr_boxes = []
        self.original_help = True
        self.redo_tr = False
        self.box_coordinate = None
        self.refine = False
        self.setContentsMargins(0, 0, 0, 0)
        
    def _init_ui(self):
        print('This message should not be seen. Please override _init_ui')

    def _imview_clicked(self, event):
        if event.button() == QtCore.Qt.RightButton:
            event.ignore()
            return

        if self.ops is None:
            return

        pos = self.imview.getImageItem().mapFromScene(event.pos())
        
        self._size_ops = self.ops.data.shape[0]*0.01
        if self.other.ops is not None:
            self._size_other = self.other.ops.data.shape[0]*0.005

        item = self.imview.getImageItem()
        
        pos.setX(pos.x() - self._size_ops/2)
        pos.setY(pos.y() - self._size_ops/2)

        if self.define_btn.isChecked():
            roi = pg.CircleROI(pos, self._size_ops, parent=item, movable=False)
            roi.setPen(255,0,0)
            roi.removeHandle(0)
            self.imview.addItem(roi)
            self.clicked_points.append(roi)
        elif self.select_btn.isChecked():
            self._draw_correlated_points(pos, self._size_ops, self._size_other, item)
        elif hasattr(self, 'select_region_btn') and self.select_region_btn.isChecked():
            '''EM only: Select individual image from montage'''
            self.box_coordinate = pos
            points_obj = (pos.x(), pos.y())

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

        if self.ops.transformed and self.other.ops.transformed:
            if self.tr_matrices is not None:
                point_obj = pg.CircleROI(pos, size1, parent=item, movable=False, removable=True)
                point_obj.setPen(0,255,0)
                point_obj.removeHandle(0)
                self.imview.addItem(point_obj)
                self.points_corr.append(point_obj)
                
                # Coordinates in clicked image
                init = np.array([point_obj.x()+size1/2,point_obj.y()+size1/2, 1])
                transf = np.dot(self.tr_matrices, init)
                pos = QtCore.QPointF(transf[0]-size2/2, transf[1]-size2/2)  
                point_other = pg.CircleROI(pos, size2, parent=self.other.imview.getImageItem(), movable=True, removable=True)
                point_other.setPen(0,255,255)
                point_other.removeHandle(0)
                self.other.imview.addItem(point_other)
                self.other.points_corr.append(point_other)
                point_obj.sigRemoveRequested.connect(lambda: self._remove_correlated_points(self.imview, self.other.imview, point_obj, point_other, self.points_corr, self.other.points_corr))
                point_other.sigRemoveRequested.connect(lambda: self._remove_correlated_points(self.other.imview, self.imview, point_other, point_obj, self.other.points_corr, self.points_corr))
        else:
            print('Transform both images before point selection')

    def _remove_correlated_points(self,imv1,imv2,pt1,pt2,pt_list,pt_list2):
        imv1.removeItem(pt1)
        imv2.removeItem(pt2)
        pt_list.remove(pt1)
        pt_list2.remove(pt2)

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
                    # If assembly is an option (EM image)
                    if hasattr(self, 'show_assembled_btn'):
                        if self.show_assembled_btn.isChecked():
                            self.ops.orig_points = points
                        else:
                            self.ops.orig_points_region = points
                    else:
                        self.ops.orig_points = points
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
            pos = [QtCore.QPointF(point[0], point[1]) for point in self.ops.points]
            poly_line = pg.PolyLineROI(pos, closed=True, movable=False)
            if self.show_btn.isChecked():     
                if self.show_grid_btn.isChecked():
                    if not toggle_orig:
                        self.imview.removeItem(self.tr_grid_box)
                        self.imview.removeItem(self.grid_box)
                        self.show_grid_box = False
                print('Recalculating original grid...')
                self.grid_box = poly_line
            else:
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
       
        check_obj = self.ops._tf_data is not None or (hasattr(self.ops, 'tf_region') and self.ops.tf_region is not None)
        check_other = self.other.ops._tf_data is not None or (hasattr(self.other.ops, 'tf_region') and self.other.ops.tf_region is not None)
        
        if check_obj and check_other:
            if checked:
                print('Select points of interest on %s image'%self.tag)
                if len(self.points_corr) != 0:
                    [self.imview.removeItem(point) for point in self.points_corr]
                    [self.other.imview.removeItem(point) for point in self.other.points_corr]
                    self.points_corr = []
                    self.other.points_corr = []
                src_sorted = np.array(sorted(self.ops.points, key=lambda k: [np.cos(30*np.pi/180)*k[0] + k[1]]))
                dst_sorted = np.array(sorted(self.other.ops.points, key=lambda k: [np.cos(30*np.pi/180)*k[0] + k[1]]))
                self.tr_matrices = self.ops.get_transform(src_sorted, dst_sorted)
            else:
                print('Done selecting points of interest on %s image'%self.tag)
        else:
            if checked:
                print('Select and transform both data first')

    def _affine_transform(self):
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
            self._recalc_grid(toggle_orig=True)
            self._update_imview()
            self.transform_btn.setEnabled(False)
        else:
            print('Define grid box on %s image first!'%self.tag)

    def _allow_rotation_only(self, checked):
        if self.ops is not None:
            if checked:
                self.ops.no_shear = True
            else:
                self.ops.no_shear = False
            
    def _show_original(self, state):
        if self.original_help:
            if self.ops is not None:
                self.ops.transformed = not self.ops.transformed
                if hasattr(self.ops, 'flipv') and not self.ops.transformed:
                    self.flipv.setEnabled(False)
                    self.fliph.setEnabled(False)
                    self.transpose.setEnabled(False)
                    self.rotate.setEnabled(False)
                elif hasattr(self.ops, 'flipv') and self.ops.transformed:
                    if not self.ops.refined:
                        self.flipv.setEnabled(True)
                        self.fliph.setEnabled(True)
                        self.transpose.setEnabled(True)
                        self.rotate.setEnabled(True)

                print('Transformed?', self.ops.transformed)
                if hasattr(self.ops,'refined'):
                    if self.ops.refined:
                        self.ops.toggle_original(update=False)
                    else:
                        self.ops.toggle_original()
                else:
                    self.ops.toggle_original()
                self._recalc_grid(toggle_orig=True)

                self._update_imview()
                if self.ops.transformed:
                    self.transform_btn.setEnabled(False)
                else:
                    self.transform_btn.setEnabled(True)

    def _refine(self):
        if len(self.points_corr) > 3:
            src = np.array([[point.x()+self._size_ops/2,point.y()+self._size_ops/2] for point in self.points_corr])
            dst = np.array([[point.x()+self._size_other/2,point.y()+self._size_other/2] for point in self.other.points_corr])
            self.ops.calc_refine_matrix(src, dst,self.other.ops.points)
            [self.imview.removeItem(point) for point in self.points_corr]
            [self.other.imview.removeItem(point) for point in self.other.points_corr]
            self.points_corr = []
            self.other.points_corr = []
            self.refine = True
            self._recalc_grid()
            self._update_imview() 
            self.fliph.setEnabled(False)
            self.flipv.setEnabled(False)
            self.transpose.setEnabled(False)
            self.rotate.setEnabled(False)
        else:
            print('Select at least 4 points for refinement!')
        

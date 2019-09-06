import numpy as np
import sys
from PyQt5 import QtWidgets, QtCore
import gui
import pyqtgraph as pg

def put_grid_points(controls, points):
    controls.define_btn.setChecked(True)
    for point in points:
        roi = pg.CircleROI(tuple(point), (5,5), parent=controls.imview.getImageItem(), movable=False)
        roi.setPen(255,0,0)
        roi.removeHandle(0)
        controls.clicked_points.append(roi)
    controls.define_btn.setChecked(False)    
    controls._affine_transform()

class Click():
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def x(self):
        return self._x

    def y(self):
        return self._y

em_fname = 'data/5b2_montage.mrc'
fm_fname = 'data/5b_2.lif'

app = QtWidgets.QApplication([])

g = gui.GUI()
fm = g.fmcontrols
em = g.emcontrols

# Set up FM image
fm._parse_fm_images(fm_fname, series=0)
fm.imview.setLevels(1.5,4)
points = np.array(
    [[ 416.98723286,  892.0407028 ],
     [ 922.20282787, 1355.78337583],
     [1359.55364146,  835.48671828],
     [ 858.10831209,  364.20351398]])
put_grid_points(fm, points)
fm._fliph(True)
fm._flipv(True)

# Set up EM image
em.mrc_fname.setText(em_fname)
em._assemble_mrc()
em._select_box(True)
em.box_coordinate = Click(470, 440)
em._select_box(False)
points = np.array(
    [[1300.14016555,  592.91417233],
     [1161.15892202,  970.68388727],
     [1551.19605629, 1093.098682  ],
     [1675.96172094,  713.42019602]])
put_grid_points(em, points)

sys.exit(app.exec_())

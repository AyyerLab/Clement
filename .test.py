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
em_fname = 'data/5b2_montage.mrc'
fm_fname = 'data/5b_2.lif'

app = QtWidgets.QApplication([])

g = gui.GUI()
g.fmcontrols._parse_fm_images(fm_fname)
g.fm_imview.setLevels(1.5,4)
g.emcontrols.mrc_fname.setText(em_fname)
g.emcontrols._assemble_mrc()

points = np.array(
    [[ 892.0407028 ,  416.98723286],
     [1355.78337583,  922.20282787],
     [ 835.48671828, 1359.55364146],
     [ 364.20351398,  858.10831209]])
put_grid_points(g.fmcontrols, points)

points = np.array(
   [[515.52975312, 417.48259942],
    [553.85672004, 429.28672231],
    [540.59529802, 467.75941914],
    [502.2683311 , 455.08091678]])
put_grid_points(g.emcontrols, points)

sys.exit(app.exec_())

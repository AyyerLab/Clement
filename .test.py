import numpy as np
import sys
from PyQt5 import QtWidgets, QtCore
import gui
import pyqtgraph as pg

app = QtWidgets.QApplication([])

g = gui.GUI()
g._parse_fm_images('data/5b_2.lif')
g.fm.parse(g.fm.old_fname, z=10)
g.fm_imview.setLevels(1.5,4)
g.mrc_fname.setText('data/5b2_montage.mrc')
g._assemble_mrc()

points = [[ 892.0407028 ,  416.98723286],
     [1355.78337583,  922.20282787],
     [ 835.48671828, 1359.55364146],
     [ 364.20351398,  858.10831209]]
g.define_btn.setChecked(True)
for point in points:
    roi = pg.CircleROI(tuple(point), (5,5), parent=g.fm_imview.getImageItem(), movable=False)
    roi.setPen(255,0,0)
    roi.removeHandle(0)
    g.clicked_points[0].append(roi)
g.define_btn.setChecked(False)    
g._affine_transform(g.fm_imview)
g.rotate.setChecked(True)
g.flipv.setChecked(True)

points = np.array(
    [[515.52975312, 417.48259942],
    [553.85672004,  429.28672231],
    [540.59529802, 467.75941914],
    [502.2683311 , 455.08091678]])
g.define_btn_em.setChecked(True)
for point in points:
    roi = pg.CircleROI(tuple(point-2.5), (5,5), parent=g.em_imview.getImageItem(), movable=False)
    roi.setPen(255,0,0)
    roi.removeHandle(0)
    g.clicked_points[1].append(roi)
g.define_btn_em.setChecked(False)    
g._affine_transform(g.em_imview)

sys.exit(app.exec_())

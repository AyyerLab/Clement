import pyqtgraph as pg
win = pg.GraphicsWindow()
vb = win.addViewBox()
img1 = pg.ImageItem(pg.np.random.normal(size=(100,100)))
img2 = pg.ImageItem(pg.np.random.normal(size=(10,10)))
vb.addItem(img1)
vb.addItem(img2)
img2.setZValue(10) # make sure this image is on top
img2.setOpacity(0.5)
img2.scale(10, 10)

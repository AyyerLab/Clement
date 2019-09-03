# Clement
## GUI for Correlative Light and Electron Microscopy
A graphical program to correlate electron microscopy and fluorescent optical images of the same sample.

## Installation from source
Here are the instructions to install the program from source and package it for distribution. In the future, binaries will also be released.

 * Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you do not already have it.
 * Clone this repository
 ```
 $ git clone https://github.com/kartikayyer/CLEMent.git
 $ cd CLEMent
 ```
 * Create a conda environment using the environment.yml file provided:
 ```
 $ conda env create -f environment.yml
 $ conda activate clem
 ```
 * Create the resources file;
 ```
 $ pyrcc5 styles/res.qrc -o res_styles.py
 ```
 * You can now run the program directly by running `./gui.py`, or you can compile to an executable:
 ```
 $ pyinstaller clement.spec
 ```
 * This will create a single file executable in the `dist/` folder which you can use directly and on other computers with compatible platforms.

## Tutorial
Instructions for semi-automatic alignment of FM and EM images based on observable features is described below.

 * Open the GUI and load the FM image stack and EM montage. Assemble the EM montage.
 ![Load images][load]
 You can zoom and pan the image using the mouse and adjust the color scale using the bars on the right of each image. For the FM image, you can also choose which color channels to show and modify the color of each channel by clicking the square with the colors.
 * Pick a slice of the FM image using the arrow keys or select the maximum value projection through the stack. For the EM image, use the 'Select subregion' button to get the relevant slice from the montage.
 ![Select relevant parts][slice]
 After the relevant box is highlighted in orange on the EM image, unselect 'Select subregion' to zoom in.
 * Click the 'Define Grid' button and select the corners of the equivalent grid squares in each image. Once the four corners have been selected, unselect 'Define Grid'.
 ![Define grid square][grid]
 A box will appear, whose corners you can drag to adjust it. In this example, we have not drawn great boxes, but that is okay. Click 'Transform Image' on both sides. This modifies the image such that the selected grids become squares. You can now use the flips and rotates to make the features on both sides match approximately. In this case, we needed a single clockwise rotation. The specific choice will depend on exactly in which order you drew the box corners.
 * Now we will refine the alignment by aligning equivalent features on either side. Here, we have used the holes in the lacey carbon film, but one could also use fluorescent beads or other markers.
 Click 'Select points of interest' in the FM side and click on some features which are clearly visible in both images. Green circles will appear where you clicked on the FM image and cyan circles will appear on the EM image which are supposedly equivalent.
 ![Select points of interest][points]
 As you can see, the points do not match very well. But the cyan points are movable, so drag them to be match the FM points as close as you can. Feel free to zoom in. You can also select more points in other parts of the image to improve precision and be less sensitive to a single point being misaligned.
 * Once you are done, uncheck 'Select points of interest' and click 'Refinement'. You can select points again to see how well the refinement worked and repeat the previous step as many times.
 ![Refined alignment][refine]
 * Once you are satisfied with the point-wise alignment, you can click 'Merge' on the FM side. This produces a popup window showing the merged image where the EM image is now shown as an extra color channel.
 ![Merged, aligned images][merge]
 You can export the merged image, and soon you will be able to select points and export their stage positions on the EM image to collect high magnification data.

[load]: images/load_images.png
[slice]: images/select_slices.png
[grid]: images/grid_transform.png
[points]: images/select_points.png
[refine]: images/refine.png
[merge]: images/merge_popup.png


from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import filters
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from sys import getsizeof
from scipy import ndimage
import skimage.measure
import seaborn as sns
import pandas as pd 
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.graph import route_through_array
import datetime
import numpy as np
from scipy.spatial import cKDTree, distance
from scipy.stats import entropy
from skimage.measure import regionprops, label
from skimage.morphology import convex_hull_image
from sklearn.neighbors import NearestNeighbors
from skimage.util import img_as_float
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import math
from scipy.spatial import KDTree
import scipy.ndimage
import joblib as jb
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def downscale_image_with_interpolation(ct_stack, scale_factor=0.5, order=1):
    # Compute the new shape
    new_shape = np.array(ct_stack.shape) * scale_factor
    new_shape = new_shape.astype(int)

    # Downscale using scipy.ndimage.zoom with interpolation
    downscaled_ct_stack = scipy.ndimage.zoom(ct_stack, (scale_factor, scale_factor, scale_factor), order=order)

    return downscaled_ct_stack

dirs = ['M1','M2','M3','M4','M5','M6','M7','M8','P1','P2','P3','P5','P6','P7','P8']

for k in range(len(dirs)):
    tiff_dir = dirs[k]
    file_names = sorted(os.listdir(tiff_dir))
    # Load all the TIFF images in the directory into a 3D array
    v_size = 0.04358306*(1/0.7)
    l_pix = np.int16(np.ceil(300/0.04358306))
    w_pix = np.int16(np.ceil(15/0.04358306))
    thresholded_ct_array = []
    image_stack = []
    for filename in sorted(file_names):
        if filename.endswith(".tiff") or filename.endswith(".tif"):
            img_path = os.path.join(tiff_dir, filename)
            img = Image.open(img_path)
            image_stack.append((np.array(img)/256).astype('uint8'))

    ct_array = np.stack(image_stack, axis=0)  # Shape: (z, y, x)
    del image_stack
    ct_array = downscale_image_with_interpolation(ct_array,0.7)
    jb.dump(ct_array,'JB/'+dirs[k]+'.dat')


for k in range(len(dirs)):
    # Load all the TIFF images in the directory into a 3D array
    v_size = 0.04358306*(1/0.7)
    l_pix = np.int16(np.ceil(300/v_size))
    w_pix = np.int16(np.ceil(15/v_size))
    thresholded_ct_array = []
    ct_array = jb.load('JB/'+dirs[k]+'.dat')

    if k not in [2,13,14]:
        print(k)
        ct_array = np.flip(ct_array,[0,1])


    rect_width, rect_height = 500, int(300/v_size)
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.imshow(ct_array[:,:,250])

    # Initialize rectangle (will be updated interactively)
    rect = Rectangle((0, 0), rect_width, rect_height, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(rect)

    # Update rectangle position on mouse move
    def on_mouse_move(event):
        if event.xdata and event.ydata:
            rect.set_xy((event.xdata - rect_width / 2, event.ydata - rect_height / 2))
            fig.canvas.draw_idle()

    # Confirm position on mouse click
    def on_click(event):
        global ct_array
        if event.xdata and event.ydata:
            x, y = int(event.xdata - rect_width / 2), int(event.ydata - rect_height / 2)
            roi = ct_array[:,:,250][y:y + rect_height, x:x + rect_width]
            plt.figure()
            plt.imshow(roi)
            plt.title("Selected ROI")
            plt.show()
            ct_array = ct_array[y:y + rect_height,:,:]

    # Connect events
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

    plt.imshow(ct_array[:,:,250])
    plt.show()
    rect_width, rect_height = 500, 500
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.imshow(ct_array[2569,:,:])

    # Initialize rectangle (will be updated interactively)
    rect = Rectangle((0, 0), rect_width, rect_height, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(rect)

    # Update rectangle position on mouse move
    def on_mouse_move(event):
        if event.xdata and event.ydata:
            rect.set_xy((event.xdata - rect_width / 2, event.ydata - rect_height / 2))
            fig.canvas.draw_idle()

    # Confirm position on mouse click
    def on_click(event):
        global ct_array
        if event.xdata and event.ydata:
            x, y = int(event.xdata - rect_width / 2), int(event.ydata - rect_height / 2)
            roi = ct_array[2569,:,:][y:y + rect_height, x:x + rect_width]
            plt.figure()
            plt.imshow(roi)
            plt.title("Selected ROI")
            plt.show()
            ct_array = ct_array[:,y:y + rect_height, x:x + rect_width]

    # Connect events
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

    plt.imshow(ct_array[2569,:,:])
    plt.show()


    jb.dump(ct_array,'Cropped/'+dirs[k]+'.dat')

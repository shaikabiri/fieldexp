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
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, morphology
from skimage.filters import threshold_otsu
from skan.csr import skeleton_to_csgraph
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import tifffile as tiff

# Assuming 'ct_stack' is your 3D image (CT stack), with shape (depth, height, width)

def fractal_dimension(binary_stack):
    # Assert binary_stack is indeed 3D
    assert len(binary_stack.shape) == 3, "Input stack must be a 3D binary array."

    # Calculate the box sizes
    sizes = np.arange(1, min(binary_stack.shape), 2)

    counts = []
    for size in sizes:
        # Resize the array into boxes of the current size
        shape = (binary_stack.shape[0] // size, size,
                 binary_stack.shape[1] // size, size,
                 binary_stack.shape[2] // size, size)
        reshaped = binary_stack[:shape[0] * size, :shape[2] * size, :shape[4] * size].reshape(shape)
        
        # Count boxes that contain part of the pore space
        non_empty_boxes = (reshaped.max(axis=(1, 3, 5)) > 0).sum()
        counts.append(non_empty_boxes)

    # Fit a line to the log-log plot to determine the fractal dimension
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def downscale_image_with_interpolation(ct_stack, scale_factor=0.5, order=1):
    # Compute the new shape
    new_shape = np.array(ct_stack.shape) * scale_factor
    new_shape = new_shape.astype(int)

    # Downscale using scipy.ndimage.zoom with interpolation
    downscaled_ct_stack = scipy.ndimage.zoom(ct_stack, (scale_factor, scale_factor, scale_factor), order=order)

    return downscaled_ct_stack

# Directory containing the TIFF images
dirs = ['M1','M2','M3','M4','M5','M6','M7','M8','P1','P2','P3','P5','P6','P7','P8']



#thresholds = [16.696180555555557, 43.09375, 25.21527777777778, 19.286458333333336, 24.364583333333332, 21.32638888888889, 20.244791666666668, 18.756944444444443, 22.63888888888889, 14.385416666666666, 15.391493055555554, 18.310763888888893, 30.38888888888889, 20.463541666666664, 34.68836805555556]

for k in range(len(dirs)):
    ct_array = jb.load('Cropped/'+dirs[k]+'.dat') 
    #ct_array = downscale_image_with_interpolation(ct_array,0.5)   
    v_size = 0.04358306*(0.7)
    for i in range(ct_array.shape[0]):
        ct_array[i,:,:] = ndimage.median_filter(ct_array[i,:,:], size=2)
        
    #ct_thresh = ct_array > np.int16(thresholds[k])
    ct_thresh = np.empty(ct_array.shape,dtype=np.uint8)
    ct_thresh2 = np.empty(ct_array.shape,dtype=np.uint8)

    for m in range(ct_array.shape[0]):
        ct_thresh[m,:,:] = ct_array[m,:,:]>filters.threshold_multiotsu(ct_array[m,:,:],4)[0]
        ct_thresh2[m,ct_array[m,:,:]<filters.threshold_multiotsu(ct_array[m,:,:],4)[0]] = 0
        ct_thresh2[m,ct_array[m,:,:]>filters.threshold_multiotsu(ct_array[m,:,:],4)[0]] = 1
        ct_thresh2[m,ct_array[m,:,:]>filters.threshold_multiotsu(ct_array[m,:,:],4)[1]] = 2
        ct_thresh2[m,ct_array[m,:,:]>filters.threshold_multiotsu(ct_array[m,:,:],4)[2]] = 3
    


    struct_element = ndimage.generate_binary_structure(3, 1)
    ct_thresh = ndimage.binary_opening(ct_thresh, structure=struct_element, iterations=2)
    #ct_thresh = ndimage.binary_closing(ct_thresh, structure=struct_element, iterations=2)

    # im1 = plt.imread('core_slice.png')

    # fig, axs = plt.subplots(2,2)
    # fig.set_figheight(10)
    # fig.set_figwidth(10)
    
    # axs[0, 0].imshow(im1,cmap='gray')
    # axs[0, 1].imshow(ct_array[2569,:,:],cmap='gray')
    # axs[1, 0].imshow(ct_thresh2[2569,:,:])
    # axs[1, 1].imshow(ct_thresh[2569,:,:],cmap='gray')
    # axs[0, 0].tick_params(left = False, right = False , labelleft = False , 
    #             labelbottom = False, bottom = False) 
    # axs[0, 1].tick_params(left = False, right = False , labelleft = False , 
    #         labelbottom = False, bottom = False) 
    # axs[1, 1].tick_params(left = False, right = False , labelleft = False , 
    #         labelbottom = False, bottom = False) 
    # axs[1, 0].tick_params(left = False, right = False , labelleft = False , 
    #         labelbottom = False, bottom = False) 
    # axs[0, 0].set_title('A',loc='left',fontsize=12)
    # axs[0, 1].set_title('B',loc='left',fontsize=12)
    # axs[1, 0].set_title('C',loc='left',fontsize=12)
    # axs[1, 1].set_title('D',loc='left',fontsize=12)
    # plt.show()

    del ct_array

    binary = np.array_split(~ct_thresh,3,axis=0)
    tiff.imwrite('tiffs/'+dirs[k]+'.tiff',binary[1])
    n_pores = []
    median_pores = []
    final_res = []


    for i in [1]:
        print(i)
        binary_stack = binary[i]
        vol = binary_stack.shape[0]*binary_stack.shape[1]*binary_stack.shape[2]*v_size * v_size * v_size * 0.001
        # Step 2: Label connected components (find individual pores)
        labeled_stack, num_features = ndimage.label(binary_stack)
        # Coordinates of all pore voxels
        pore_coords = np.argwhere(binary_stack == 1)
        pore_count = len(pore_coords)

        # Step 3: Measure properties of labeled regions (e.g., volume of each pore)
        pore_props = skimage.measure.regionprops(labeled_stack)

        # Store the estimated pore sizes
        pore_sizes = []
        pore_volumes = []
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for prop in pore_props:
            # Calculate the pore size (area for 2D, volume for 3D)
            # Multiply by voxel size to convert to physical units (optional)
            pore_volume = prop.area * v_size * v_size * v_size
            pore_volumes.append(pore_volume)
            if pore_volume > 0:
                pore_radius = (3 * pore_volume / (4 * np.pi)) ** (1/3)
                pore_sizes.append(pore_radius*2)
                if np.logical_and(pore_radius*2<2,pore_radius*2>5*v_size):
                    ax.scatter(prop.centroid[0],prop.centroid[1],prop.centroid[2], c='red', marker='x',s=1)
        ax.set_box_aspect((1, 1, 3))  
        ax.grid(False)
        ax.set_zticks([])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_facecolor("black")

        ax._axis3don = False

        x_min = 0
        y_min = 0
        z_min = 0 
        x_max = binary_stack.shape[0]
        y_max = binary_stack.shape[1]
        z_max = binary_stack.shape[2]
        # Draw the bounding box (outline)
        # Bottom edges
        ax.plot([x_min, x_max], [y_min, y_min], [z_min, z_min], c='white', lw=1)
        ax.plot([x_min, x_max], [y_max, y_max], [z_min, z_min], c='white', lw=1)
        ax.plot([x_min, x_min], [y_min, y_max], [z_min, z_min], c='white', lw=1)
        ax.plot([x_max, x_max], [y_min, y_max], [z_min, z_min], c='white', lw=1)

        # Top edges
        ax.plot([x_min, x_max], [y_min, y_min], [z_max, z_max], c='white', lw=1)
        ax.plot([x_min, x_max], [y_max, y_max], [z_max, z_max], c='white', lw=1)
        ax.plot([x_min, x_min], [y_min, y_max], [z_max, z_max], c='white', lw=1)
        ax.plot([x_max, x_max], [y_min, y_max], [z_max, z_max], c='white', lw=1)

        # Vertical edges (connecting bottom and top)
        ax.plot([x_min, x_min], [y_min, y_min], [z_min, z_max], c='white', lw=1)
        ax.plot([x_min, x_min], [y_max, y_max], [z_min, z_max], c='white', lw=1)
        ax.plot([x_max, x_max], [y_min, y_min], [z_min, z_max], c='white', lw=1)
        ax.plot([x_max, x_max], [y_max, y_max], [z_min, z_max], c='white', lw=1)

        # Set the aspect ratio manually
        ax.set_box_aspect((1, 1, 3))  # Aspect ratio for x:y:z


        # Set the aspect ratio manually
        ax.set_box_aspect((1, 1, 3))  # Aspect ratio for x:y:z
        plt.savefig('images/'+dirs[k]+'.jpg', dpi=300, )
        plt.show()

        pore_sizes = np.array(pore_sizes)
        pore_volumes = np.array(pore_volumes)        
        


        final_res.append([dirs[k], i, np.sum(binary_stack)/(binary_stack.shape[0]*binary_stack.shape[1]*binary_stack.shape[2]),
                        np.sum(pore_volumes[pore_sizes<1])*0.001/vol,
                        np.sum(pore_volumes[np.logical_and(pore_sizes<1,pore_sizes>2*v_size)])*0.001/vol,
                        len(pore_sizes[pore_sizes<1])/vol,
                        len(pore_sizes[pore_sizes>2*v_size])/vol,
                        len(pore_sizes[pore_sizes>1])/vol,
                        np.nanmean(pore_sizes),np.nanmedian(pore_sizes),
                        np.max(pore_sizes),np.min(pore_sizes)])
    print(final_res)
    final_res = pd.DataFrame(final_res)
    final_res.to_csv('Res/'+dirs[k]+'.csv',index=False,header=False)
    print('end')
    print(datetime.datetime.now().time())


source_files = sorted(os.listdir('res'))
dataframes = []
for file in source_files:
    df = pd.read_csv('res/'+file,header=None) 
    dataframes.append(df)

df_all = pd.concat(dataframes)
df_all.columns = ['core','depth', 'porosity', 'macroporosity', 'macroporosity2' , 'macropore_density', 'macropore_density2','megapore_densityy','mean_pore_diameter','median_pore_diameter','max_pore_diameter', 'min_pore_diameter','frac_dim']
df_all.to_csv('res_final_4phase.csv',index=None)

source_files = sorted(os.listdir('res'))
dataframes = []
for file in source_files:
    df = pd.read_csv('res/'+file,header=None) 
    dataframes.append(df)

df_all = pd.concat(dataframes)
df_all.columns = ['core','depth', 'porosity', 'macroporosity' , 'macropore_density','megapore_densityy','mean_pore_diameter','median_pore_diameter','max_pore_diameter', 'min_pore_diameter','frac_dim','ent']
df_all.to_csv('res_final.csv',index=None)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from skimage import io, img_as_float

# Load your image (replace with your file path)
# Example uses a synthetic grayscale image. Replace 'your_image.tif' with your XCT image path.
  # Change this to your XCT image file path
image = ct_array[500:1000,200,:]  # Read and normalize image to range [0, 1]

# Function to apply threshold and update the display
def update_threshold(val):
    global selected_threshold
    selected_threshold = slider.val  # Update the global variable
    binary = image > selected_threshold
    ax_binary.imshow(binary, cmap='gray')
    fig.canvas.draw_idle()

# Function to confirm and store the selected threshold
def confirm_threshold(event):
    global selected_threshold
    print(f"Threshold confirmed: {selected_threshold}")
    # The threshold value is now stored in 'selected_threshold' and can be used later

thresholds = []
for k in range(len(dirs)):
    ct_array = jb.load('Cropped/'+dirs[k]+'.dat')    
    v_size = 0.04358306*(1/0.7)
    for i in range(ct_array.shape[0]):
        ct_array[i,:,:] = ndimage.median_filter(ct_array[i,:,:], size=2)

    image = ct_array[500:1000,200,:] 

    # Create the figure and axes
    fig, (ax_original, ax_binary) = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(bottom=0.2)

    # Display the original image
    ax_original.set_title('Original Image')
    ax_original.imshow(image, cmap='gray')
    ax_original.axis('off')

    # Display the initial binary image
    ax_binary.set_title('Binary Image')
    binary_initial = image > 0.5  # Initial threshold value
    ax_binary.imshow(binary_initial, cmap='gray')
    ax_binary.axis('off')

    # Create the slider for threshold adjustment
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])  # Position of the slider
    slider = Slider(ax_slider, 'Threshold', np.min(image), np.max(image), valinit=0.5)

    # Create the button to confirm the threshold value
    ax_button = plt.axes([0.8, 0.02, 0.1, 0.04])  # Position of the button
    button = Button(ax_button, 'Confirm')

    # Connect the slider to the update function
    slider.on_changed(update_threshold)


    # Connect the slider to the update function
    slider.on_changed(update_threshold)

    # Connect the button to the confirm function
    button.on_clicked(confirm_threshold)

    plt.show()
    
    thresholds.append(selected_threshold)
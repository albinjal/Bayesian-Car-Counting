from scipy.io import loadmat,savemat
from PIL import Image, ImageOps
import numpy as np
import os
import cv2
# pip install gimpformats
from gimpformats.gimpXcfDocument import GimpDocument

# im_path = '12TVK220980-CROP_Annotated.xcf'
def convertToMat(im_path):
  project = GimpDocument(im_path)
  layers = project.layers
  # Convert the image to grayscale then to a NumPy array
  image_array = np.array(ImageOps.grayscale(layers[0].image))

  # Find the coordinates of pixels with values greater than 0
  coordinates = np.argwhere(image_array > 0)
  mat_path = im_path.replace('_Annotated.xcf', '_ann.mat') 
  savemat(mat_path, {'annPoints': coordinates})

import glob

folder_path = "CWOC_mat/"
xcf_files = glob.glob(folder_path + "*.xcf")

for file in xcf_files:
    print(file)
    convertToMat(file)

from scipy.io import loadmat,savemat
from PIL import Image, ImageOps
import numpy as np
import os
import cv2
# pip install gimpformats
from gimpformats.gimpXcfDocument import GimpDocument

def split_image_into_patches(image, patch_size):
    width, height = image.size
    patch_width, patch_height = patch_size

    patches = []
    for y in range(0, height, patch_height):
        for x in range(0, width, patch_width):
            if x + patch_width > width:
                x = width - patch_width
            if y + patch_height > height:
                y = height - patch_height
            box = (x, y, x + patch_width, y + patch_height)
            patch = image.crop(box)
            patches.append(patch)
    return patches

# im_path = '12TVK220980-CROP_Annotated.xcf'
def convertToMatPatches(im_path,image_number,patch_size):
  project = GimpDocument(im_path)
  layers = project.layers
  annotation = None
  image = None
  count = 0
  for layer in layers:
    if "Car" in layer.name:
      annotation=ImageOps.grayscale(layer.image)
    if "Car" not in layer.name or "Neg" not in layer.name:
      image=layer.image

  image_patches = split_image_into_patches(image, patch_size)
  annotation_patches = split_image_into_patches(annotation, patch_size)
  # Save the patches or perform any further operations

  for i in range(len(image_patches)):
    print(i)
    name=f'patches/patch_{image_number}_{i}.jpg'
    # Find the coordinates of pixels with values greater than 0
    coordinates = np.argwhere(np.array(annotation_patches[i]) > 0)
    # if there is atkeast one car save the patch
    if len(coordinates)>0:
      image_patches[i].convert('RGB').save(name)
      annotation_patches[i].save(name.replace('.jpg', '_ann.png'))
      mat_path = name.replace('.jpg', '_ann.mat') 
      savemat(mat_path, {'annPoints': coordinates})

import glob

folder_path = "xcfs/"
xcf_files = glob.glob(folder_path + "*.xcf")

patch_size = (1500, 1500)

for i,file in enumerate(xcf_files):
    print(file)
    convertToMatPatches(file,i,patch_size)

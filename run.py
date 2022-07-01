from glob import glob 
from PIL import Image
import pickle
import cv2
import numpy as np
from util import *
import jax.numpy as jnp
import numpy as np 


sub_dirs = glob("data/*/", recursive = True)
size = 224, 224  # all images will be resized to this 
new_size = (224, 224)
data_set = []

for subdir in sub_dirs:
	new_subdirs = sorted(glob(subdir + "/*/", recursive = True))

	# image 
	im_files = glob(new_subdirs[0] + '/*.' + 'png')
	img_file = im_files[0]
	image = cv2.imread(img_file)  # 0 means grayscale
	image = cv2.resize(image, new_size)
	image_array = np.array(image)
	image_array = image_array.astype(float)
	image_array = image_array/255.0
	jax_img_array = jnp.array(image_array)

	# mask 
	mask_files = glob(new_subdirs[1] + '/*.' + 'png')
	mask_arrays = []
	spliced_segment = np.zeros((224, 224 , 1), dtype=float)
	for elem in mask_files:
		img_segment = cv2.imread(elem, 0)  # 0 means grayscale
		img_segment = cv2.resize(img_segment, new_size)
		img_segment = np.expand_dims(img_segment, axis=-1)
		# overlay the mask 
		spliced_segment = np.maximum(spliced_segment, img_segment)

	# binarize the mask (only 0 and 1 values)
	overlayed_mask = (spliced_segment > 0).astype(float)

	# jax-ify, tuple, and append 
	jax_mask_array = jnp.array(overlayed_mask)
	sample_tup = (jax_img_array, jax_mask_array)
	data_set.append(sample_tup)

	# # plot to visualize 
	# plot_numpy_img(jax_img_array, True)
	# plot_numpy_img(jax_mask_array, False)

# pickle up dataset 
out_file = open("jax_cell_data.pkl", "wb")
pickle.dump(data_set, out_file)
out_file.close()

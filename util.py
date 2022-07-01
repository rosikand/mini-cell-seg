"""
File: util.py 
-------------- 
Contains useful functions that I may import while programming
for tasks at insitro. 

To-do:
1. image to PyTorch tensor function 
""" 

from matplotlib import pyplot as plt
from PIL import Image
import jax.numpy as jnp
import numpy as np 
from glob import glob
import random


def plot_png(file_path, color):
	"""
	Plots a png file (or jpeg really) using matplotlib. Color is a bool value. 
	If true, image will be displayed in color. Else, it will be displayed in 
	black and white.
	"""
	image = Image.open(file_path)
	if color == True:
		plt.imshow(image, interpolation='nearest')
		plt.show()
	else:
		plt.imshow(image, interpolation='nearest', cmap='Greys_r')
		plt.show()


def plot_numpy_img(image_array, color):
	"""
	Plots a numpy array using matplotlib. Color is a bool value. If true, 
	image will be displayed in color. Else, it will be displayed in 
	black and white. 
	"""
	if color == True:
		plt.imshow(image_array, interpolation='nearest')
		plt.show()
	else:
		plt.imshow(image_array, interpolation='nearest', cmap='Greys_r')
		plt.show()


def plot_tensor(input_tensor, color):
    """
    plots the PyTorch tensor input_tensor
    with the corresponding color map. 
    """
    input_tensor = torch.squeeze(input_tensor)
    np_arr = input_tensor.numpy()
    plot_numpy_img(np_arr, color)


def image_dir_to_data(dirpath, extension):
    """
    Takes in a directory containing images
    and returns a list of numpy arrays representing
    those images. 
    Args:
        - dirpath: path to directory 
        - extension: image extension type (e.g. png) (string)
    """
    data_subset = []
    sub_set = glob(dirpath + '/*.' + extension)
    for elem in sub_set:
        image = Image.open(elem)
        image_array = np.array(image)
        data_subset.append(image_array)
    return data_subset  


def image_dir_to_jax_data(dirpath, extension):
    """
    Same as image_dir_to_data but exports to jax np arrays
    instead of traditional np arrays.  
    Args:
        - dirpath: path to directory 
        - extension: image extension type (e.g. png) (string)
    """
    data_subset = []
    sub_set = glob(dirpath + '/*.' + extension)
    for elem in sub_set:
        image = Image.open(elem)
        image_array = np.array(image)
        jax_array = jnp.array(image_array)
        data_subset.append(jax_array)
    return data_subset     


def image_dir_to_jax_data_norm(dirpath, extension):
    """
    Same as image_dir_to_jax_data but adds /255 image normalization 
    and converts to image array to type float.  
    Args:
        - dirpath: path to directory 
        - extension: image extension type (e.g. png) (string)
    """
    data_subset = []
    sub_set = glob(dirpath + '/*.' + extension)
    for elem in sub_set:
        image = Image.open(elem)
        image_array = np.array(image)
        image_array = image_array.astype(float)
        image_array = image_array/255.0
        jax_array = jnp.array(image_array)
        data_subset.append(jax_array)
    return data_subset    

def gen_pair(data_samples, label):
	"""
	Takes in a list of data samples (i.e. images, x)
	and a corresponding label and returns a list of tuples
	where each tuple is an (x, y) pair. 
	"""
	d_set = []
	for elem in data_samples:
	    sample_pair = (elem, label)
	    d_set.append(sample_pair)
	return d_set 


def gen_distro(master_list):
    """
    Takes in a list of lists who each contain
    (x, y) tuple samples for that class and returns 
    a randomly concatenated version. 
    """
    generated_distro = []
    for elem in master_list:
        generated_distro = generated_distro + elem
    random.shuffle(generated_distro)
    return generated_distro


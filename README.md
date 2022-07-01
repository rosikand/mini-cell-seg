# `mini-cell-seg`

This repo contains a dataset already pre-processed to test semantic segmentation models. The original data is from [here](https://www.kaggle.com/c/data-science-bowl-2018) is of nuclei in cells. Furthermore, the dataset is quite small with only 13 samples (< 21 mb). 

The original data is stored in `data.zip` and the processing script is `run.py` (which uses several functions from `util.py`). The processed dataset is just a list of arrays serialized in `.pkl` files. You can reproduce the pickled files by unzipping the `data.zip` file and running `$ python3 run.py`. 

## Dataset list format 

Data is a list of the form `[(x,y), (x,y),..., (x,y)]` where each `x` is normalized float array representing the image and each `y` is a normalized float array representing the segmentation mask. 


## Dataset files

- `np_cell_data.pkl`: dataset where each sample (image and mask) are float numpy arrays. 
- `jax_cell_data.pkl`: same as the numpy version except the arrays are jaxlib arrays. Reduces size by 10 mb!
- `jax_cell_data_same_shape.pkl`: same as above but the images and masks have same shape and size. Originally, the images are of shape `(224, 224, 3)` (RGB) and the masks are of shape `(224, 224, 1)` (grayscale). This dataset file loads in the images in grayscale format making the shape the same as the mask `(224, 224, 1)`. This makes it convenient when using e.g., loss functions. 
- `small_set.pkl`: images and masks are downsized to `(1, 28, 28)`. Resolution is poor but less compute power is needed. Run `small_run.py` to build this. 

![image](https://user-images.githubusercontent.com/57341225/176973764-eca2160f-3d43-43bb-b2f2-c71624463aff.png)

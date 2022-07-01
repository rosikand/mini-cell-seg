# `mini-cell-seg`

This repo contains a dataset already pre-processed to test medical image segmentation models. The original data is from [here](https://www.kaggle.com/c/data-science-bowl-2018) is of nuclei in cells. Furthermore, the dataset is quite small with only 13 samples (< 21 mb). 

The original data is stored in `data.zip` and the processing script is `run.py` (which uses several functions from `util.py`). The processed dataset is just a list of arrays serialized in `.pkl` files. You can reproduce the pickled files by unzipping the `data.zip` file and running `$ python3 run.py`. 

## Dataset list format 

Data is a list of the form `[(x,y), (x,y),..., (x,y)]` where each `x` is normalized float array representing the image and each `y` is a normalized float array representing the segmentation mask. 


## Dataset files

- `np_cell_data.pkl`: dataset where each sample (image and mask) are float numpy arrays. 
- `jax_cell_data.pkl`: same as the numpy version except the arrays are jaxlib arrays. Reduces size by 10 mb!

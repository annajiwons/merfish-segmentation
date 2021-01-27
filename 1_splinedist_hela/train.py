import sys # TODO: fix hacky imports
sys.path.append('../')
sys.path.append('../splinedistm')

import numpy as np

from csbdeep.utils import normalize
from glob import glob
from skimage import io
from splinedist.models import Config2D, SplineDist2D
from splinedist.utils import phi_generator, grid_generator, get_contoursize_max
from tqdm import tqdm

from utils import train_validation_split

# Directories
image_dir = f'../images'
hela_dir = f'{image_dir}/fluo-ND2L-HeLa'
model_dir = './models'

# Read train images and masks, normalize train images
hela_X_train_names = sorted(glob(f'{hela_dir}/train/images/*.tif'))
hela_X_train = [normalize(img, 1, 99.8, axis=(0, 1)) for img in tqdm(list(map(io.imread, hela_X_train_names)))]
hela_Y_train_names = sorted(glob(f'{hela_dir}/train/masks/*.tif'))
hela_Y_train = list(map(io.imread, hela_Y_train_names))

hela_X_train, hela_Y_train, hela_X_valid, hela_Y_valid = train_validation_split(hela_X_train, hela_Y_train)

# Using configuration from SplineDist example
# choose the number of control points (M)
M = 6
n_params = 2 * M

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (2,2)

# compute the size of the largest contour present in the image-set
contoursize_max = get_contoursize_max(hela_Y_train)

conf = Config2D (
    n_params        = n_params,
    grid            = grid,
    n_channel_in    = 1,
    contoursize_max = contoursize_max,
)
print(conf)
vars(conf)

phi_generator(M, conf.contoursize_max)
grid_generator(M, conf.train_patch_size, conf.grid)

# Train SplineDist model
splinedist_hela_model = SplineDist2D(conf, name='splinedist_hela', basedir=model_dir)
splinedist_hela_model.train(hela_X_train, hela_Y_train, validation_data=(hela_X_valid, hela_Y_valid), augmenter=None, epochs=300) 
import sys
sys.path.append('../')
sys.path.append('../splinedistm')

from csbdeep.utils import normalize
from glob import glob
from skimage import io
from splinedist import fill_label_holes
from splinedist.models import Config2D, SplineDist2D
from splinedist.utils import phi_generator, grid_generator, get_contoursize_max
from tqdm import tqdm

from utils import train_validation_split

image_dir = f'/storage/annajiwons/images'
merfish_dir = f'{image_dir}/merfish'

model_dir = './models'

# Read train images that manually segmentated and fill small label holes
merfish_X_train_names = sorted(glob(f'{merfish_dir}/train/images/*.tif'))
merfish_X_train = [normalize(img, 1, 99.8, axis=(0, 1)) for img in list(map(io.imread, merfish_X_train_names))]
merfish_Y_train_names = sorted(glob(f'{merfish_dir}/train/masks/manual/*.tif'))
merfish_Y_train = [fill_label_holes(img) for img in tqdm(list(map(io.imread, merfish_Y_train_names)))]

# Train validation split
merfish_X_train, merfish_Y_train, merfish_X_valid, merfish_Y_valid = train_validation_split(merfish_X_train, merfish_Y_train)

# Using configuration from SplineDist example
# choose the number of control points (M)
M = 6
n_params = 2 * M

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (2,2)

# compute the size of the largest contour present in the image-set
contoursize_max = get_contoursize_max(merfish_Y_train)

conf = Config2D (
    n_params        = n_params,
    grid            = grid,
    n_channel_in    = 1,
    contoursize_max = contoursize_max,
    train_patch_size = (256,256),
)
print(conf)
vars(conf)

phi_generator(M, conf.contoursize_max)
grid_generator(M, conf.train_patch_size, conf.grid)

splinedist_merfish_model = SplineDist2D(conf, name='splinedist_merfish', basedir=model_dir)
splinedist_merfish_model.train(merfish_X_train, merfish_Y_train, validation_data=(merfish_X_valid, merfish_Y_valid), augmenter=None, epochs=300)

# Comment out as the default values work better
# splinedist_merfish_model.optimize_thresholds(merfish_X_valid, merfish_Y_valid)
import sys # TODO: fix hacky imports
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

# Directories
image_dir = f'../images'
merfish_dir = f'{image_dir}/merfish'

model_dir = './models'

# Read train images and normalize
merfish_X_train_names = sorted(glob(f'{merfish_dir}/train/images/*.tif'))
merfish_X_train = [normalize(img, 1, 99.8, axis=(0, 1)) for img in tqdm(list(map(io.imread, merfish_X_train_names)))]
merfish_X_train_dict = {merfish_X_train_names[i][-7:]: merfish_X_train[i] for i in range(len(merfish_X_train_names))}

# Read train masks that have been auto segmentated only and fill small label holes
merfish_Y_auto_train_names = sorted(glob(f'{merfish_dir}/train/masks/auto/*.tif'))
merfish_Y_auto_train = [fill_label_holes(img) for img in tqdm(list(map(io.imread, merfish_Y_auto_train_names)))]

auto_seg_img_names = [img_name[-7:] for img_name in merfish_Y_auto_train_names]
merfish_X_auto_train = []
for img_name in merfish_X_train_dict:
    if img_name in auto_seg_img_names:
        merfish_X_auto_train.append(merfish_X_train_dict[img_name])

# Train validation split
merfish_X_auto_train, merfish_Y_auto_train, merfish_X_auto_valid, merfish_Y_auto_valid = train_validation_split(merfish_X_auto_train, merfish_Y_auto_train)

# Using configuration from SplineDist example
# choose the number of control points (M)
M = 6
n_params = 2 * M

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (2,2)

# compute the size of the largest contour present in the image-set
contoursize_max = get_contoursize_max(merfish_Y_auto_train)

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
auto_mask_merfish_splinedist_model = SplineDist2D(conf, name='auto_mask_merfish_splinedist_model', basedir=model_dir)
auto_mask_merfish_splinedist_model.train(merfish_X_auto_train, merfish_Y_auto_train, validation_data=(merfish_X_auto_valid, merfish_Y_auto_valid), augmenter=None, epochs=300) # Didn't finish training, 283/300 epochs
import sys 
sys.path.append('../')

from csbdeep.utils import move_image_axes, normalize
from glob import glob
from skimage import io
from stardist import fill_label_holes
from stardist.matching import matching_dataset
from stardist.models import Config2D, StarDist2D
from tqdm import tqdm

from utils import train_validation_split

# Directories
image_dir = f'/storage/annajiwons/images'
mask_dir = f'{image_dir}/merfish'
input_dir = f'{image_dir}/probe_and_nuclei'

model_dir = './models'

# Read smfish train images and normalize
X_train_names = sorted(glob(f'{input_dir}/train/*.tif'))
X_train = [normalize(img, 1, 99.8, axis=(0, 1)) for img in tqdm(list(map(io.imread, X_train_names)))]
print(X_train[0].shape)

X_train = [move_image_axes(x, 'CYX', 'YXC') for x in X_train]
print(X_train[0].shape)

# Read nuclei train masks and fill small label holes
Y_train_names = sorted(glob(f'{mask_dir}/train/masks/manual/*.tif'))
Y_train = [fill_label_holes(img) for img in tqdm(list(map(io.imread, Y_train_names)))]

print(Y_train[0].shape)

# Train validation split
X_train, Y_train, X_valid, Y_valid = train_validation_split(X_train, Y_train)

conf = Config2D (
    n_rays       = 32,
    grid         = (2,2),
    n_channel_in = 2,
)
print(conf)
vars(conf)

# Train StarDist model
probe_and_nuclei_model = StarDist2D(conf, name='probe_and_nuclei_model', basedir=model_dir)
probe_and_nuclei_model.train(X_train, Y_train, validation_data=(X_valid, Y_valid), augmenter=None, epochs=300)

# Tune prob_thresh for provided (fixed) nms_thresh to maximize matching score
probe_and_nuclei_model.optimize_thresholds(X_valid, Y_valid)
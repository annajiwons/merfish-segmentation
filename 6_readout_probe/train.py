import sys 
sys.path.append('../')

from csbdeep.utils import normalize
from glob import glob
from skimage import io
from stardist import fill_label_holes
from stardist.matching import matching_dataset
from stardist.models import Config2D, StarDist2D
from tqdm import tqdm

from utils import train_validation_split

# Directories
image_dir = f'/storage/annajiwons/images'
merfish_dir = f'{image_dir}/merfish'
smfish_dir = f'{image_dir}/smfish'

model_dir = './models'

# Read smfish train images and normalize
X_train_names = sorted(glob(f'{smfish_dir}/train/*.tif'))
X_train = [normalize(img, 1, 99.8, axis=(0, 1)) for img in tqdm(list(map(io.imread, X_train_names)))]

print(X_train_names)

# Read nuclei train masks and fill small label holes
Y_train_names = sorted(glob(f'{merfish_dir}/train/masks/manual/*.tif'))
Y_train = [fill_label_holes(img) for img in tqdm(list(map(io.imread, Y_train_names)))]

print(Y_train_names)

# Train validation split
X_train, Y_train, X_valid, Y_valid = train_validation_split(X_train, Y_train)

conf = Config2D (
    n_rays       = 32,
    grid         = (2,2),
    n_channel_in = 1,
)
print(conf)
vars(conf)

# Train StarDist model
smfish_nuclei_model = StarDist2D(conf, name='readout_probe_model', basedir=model_dir)
smfish_nuclei_model.train(X_train, Y_train, validation_data=(X_valid, Y_valid), augmenter=None, epochs=300)

# Tune prob_thresh for provided (fixed) nms_thresh to maximize matching score
smfish_nuclei_model.optimize_thresholds(X_valid, Y_valid)
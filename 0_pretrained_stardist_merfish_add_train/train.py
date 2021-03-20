import sys
sys.path.append('../')

from csbdeep.utils import normalize
from glob import glob
from skimage import io
from stardist import fill_label_holes, random_label_cmap
from stardist.matching import matching_dataset
from stardist.models import Config2D, StarDist2D
from tqdm import tqdm

from utils import train_validation_split

image_dir = f'/storage/annajiwons/images'
merfish_dir = f'{image_dir}/merfish'

model_dir = './models'

# Read train images that manually segmentated and fill small label holes
merfish_X_manual_train_names = sorted(glob(f'{merfish_dir}/train/images/*.tif'))
merfish_X_manual_train = [normalize(img, 1, 99.8, axis=(0, 1)) for img in list(map(io.imread, merfish_X_manual_train_names))]
merfish_Y_manual_train_names = sorted(glob(f'{merfish_dir}/train/masks/manual/*.tif'))
merfish_Y_manual_train = [fill_label_holes(img) for img in tqdm(list(map(io.imread, merfish_Y_manual_train_names)))]

# Train validation split
merfish_X_manual_train, merfish_Y_manual_train, merfish_X_manual_valid, merfish_Y_manual_valid = train_validation_split(merfish_X_manual_train, merfish_Y_manual_train)

conf = Config2D (
    n_rays       = 32,
    grid         = (2,2),
    n_channel_in = 1,
)
print(conf)
vars(conf)

# Number of training examples to use
num_train = [5, 10]

for num in num_train:
    # Load pretrained StarDist model for 2d fluorescent images
    pretrained_stardist_model = StarDist2D(None, name = f'2D_versatile_fluo_{num}', basedir = model_dir) 
#     pretrained_stardist_model.train(merfish_X_manual_train[:num], merfish_Y_manual_train[:num], 
#                                     validation_data=(merfish_X_manual_valid[:num], merfish_Y_manual_valid[:num]), 
#                                     augmenter=None, epochs=300)
    pretrained_stardist_model.optimize_thresholds(merfish_X_manual_valid, merfish_Y_manual_valid)
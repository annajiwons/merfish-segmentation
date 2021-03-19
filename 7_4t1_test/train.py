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
nuclei_4t1_dir = f'{image_dir}/4t1'

model_dir = './models'

# Read train images that manually segmentated and fill small label holes
X_train_names = sorted(glob(f'{nuclei_4t1_dir}/train/images/*.tif'))
X_train = [normalize(img, 1, 99.8, axis=(0, 1)) for img in list(map(io.imread, X_train_names))]
Y_train_names = sorted(glob(f'{nuclei_4t1_dir}/train/masks/*.tif'))
Y_train = [fill_label_holes(img) for img in tqdm(list(map(io.imread, Y_train_names)))]

# Train validation split
X_train, Y_train, X_valid, Y_valid = train_validation_split(X_train, Y_train)

conf = Config2D (
    n_rays       = 32,
    grid         = (2,2),
    n_channel_in = 1,
)
print(conf)
vars(conf)

# Load pretrained StarDist model trained with 2d fluorescent images + 10 additional u2os images
# Train with all additional images
pretrained_stardist_model = StarDist2D(None, name = '2D_versatile_fluo_10_u2os_5', basedir = model_dir) 
pretrained_stardist_model.train(X_train, Y_train, 
                                validation_data=(X_valid, Y_valid), 
                                augmenter=None, epochs=300)
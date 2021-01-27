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

conf = Config2D (
    n_rays       = 32,
    grid         = (2,2),
    n_channel_in = 1,
)
print(conf)
vars(conf)

# Train StarDist model
auto_mask_merfish_model = StarDist2D(conf, name='auto_mask_merfish_model', basedir='models')
auto_mask_merfish_model.train(merfish_X_auto_train, merfish_Y_auto_train, validation_data=(merfish_X_auto_valid, merfish_Y_auto_valid), augmenter=None, epochs=300)
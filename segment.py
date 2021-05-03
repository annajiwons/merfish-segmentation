import argparse
import os

from csbdeep.utils import normalize
from glob import glob
from skimage import io
from stardist.models import StarDist2D
from tqdm import tqdm

def load_images(img_dir, img_ext):
    # Read images, normalize images
    print('Loading input images...')
    img_names = sorted(glob(f'{img_dir}/*.{img_ext}'))
    if len(img_names) == 0:
        print(f'No images were able to be loaded from: {img_dir}.')
        return
    imgs = [normalize(img, 1, 99.8, axis=(0, 1)) for img in tqdm(list(map(io.imread, img_names)))]
    return img_names, imgs

def main(model_num, img_dir, img_ext, out_dir):
    try:
        model_num_int = int(model_num)
    except ValueError:
        print('Please enter an integer between 1-13 for model number')
        return
    
    if model_num_int < 1 or model_num_int > 13:
        print('Please enter an integer between 1-13 for model number')
        return

    img_names, imgs = load_images(img_dir, img_ext)
    if len(imgs) == 0:
        print(f'No images loaded.')
        return
    
    # Load model
    if model_num == '1':
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
    else:
        model = StarDist2D(None, name = f'model{model_num}', basedir = './models')

    print('Segmenting nuclei...')
    for i in tqdm(range(len(imgs))):
        img_name = os.path.basename(img_names[i]).split('.')[0] # Get image name without extension
        img = imgs[i]
        seg = model.predict_instances(img, n_tiles = model._guess_n_tiles(img), show_tile_progress = False)
        io.imsave(f'{out_dir}/SEG_{img_name}.tif', seg[0]) # The first element in the result array is the image

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()    
    parser.add_argument('--model_num', required = True) # The model number to use
    parser.add_argument('--img_dir', required = True)   # Input image directory
    parser.add_argument('--img_ext', default = 'tif')  # Image extension
    parser.add_argument('--out_dir', default = './out') # Output directory 
    args = vars(parser.parse_args()) 
    main(args['model_num'], args['img_dir'], args['img_ext'], args['out_dir'])

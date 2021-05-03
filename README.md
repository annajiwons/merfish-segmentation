# merfish-segmentation

Script usage: segment.py [-h] --model_num MODEL_NUM --img_dir IMG_DIR [--img_ext IMG_EXT] [--out_dir OUT_DIR]

For --model_num, choose one of the following:
* 1:  StarDist Pretrained model trained with a subset of 2D fluorescent images from the Kaggle 2018 Data Science Bowl dataset with no additional training
* 2:  StarDist Model 1 trained with 5 additional U2OS nuclei images
* 3:  StarDist Model 1 trained with 10 additional U2OS nuclei images
* 4:  StarDist 5 U2OS nuclei training images, same as Model 2
* 5:  StarDist 10 U2OS nuclei training images, same as Model 3
* 6:  StarDist All 26 U2OS nuclei training images
* 7:  StarDist All 26 U2OS nuclei training images Random flip/rotation
* 8:  StarDist All 26 U2OS nuclei training images Random intensity
change
* 9:  StarDist All 26 U2OS nuclei training images Random Gaussian
noise
* 10: StarDist 26 U2OS readout probe channel training images
* 11: StarDist 26 U2OS 2-channel images with nuclei and readout
probe channels
* 12: StarDist Model 3 trained with 7 4t1 nuclei training images
* 13: SplineDist All 26 U2OS nuclei training images
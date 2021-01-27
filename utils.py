import matplotlib.pyplot as plt
from stardist import random_label_cmap

lbl_cmap = random_label_cmap()
taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def show_test_images(X, Y_truth, Y_pred, file_names, show_num = None):
    num_images = min(show_num, len(X)) if show_num is not None else len(X)
    for i in range(num_images):
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))
        axes[0].imshow(X[i], clim=(0,1), cmap='gray')
        axes[0].set_title(f'{file_names[i]}: raw image')
        axes[0].axis('off')
        axes[1].imshow(Y_truth[i], clim=(0,1), cmap='gray')
        axes[1].set_title(f'{file_names[i]}: ground truth')
        axes[1].axis('off')
        axes[2].imshow(X[i], clim=(0,1), cmap='gray')
        axes[2].imshow(Y_pred[i][0], cmap=lbl_cmap, interpolation="nearest", alpha=0.5)
        num_objects = len(Y_pred[i][1]['coord'])
        axes[2].set_title(f'{file_names[i]}: predicted, {num_objects} objects')
        axes[2].axis('off')
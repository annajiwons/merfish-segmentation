import matplotlib.pyplot as plt
import numpy as np

from stardist import random_label_cmap

taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def show_test_images(X, Y_truth, Y_pred, file_names, show_num = None):
    num_images = min(show_num, len(X)) if show_num is not None else len(X)
    lbl_cmap = random_label_cmap()

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

def train_validation_split(X, Y, percent_valid = 0.25):
    rng = np.random.default_rng(591)
    indices = rng.permutation(len(X))
    split_val = max(1, int(round(percent_valid * len(indices))))
    train_indices, valid_indices = indices[split_val:], indices[:split_val]
    X_train, X_valid = [X[i] for i in train_indices], [X[i] for i in valid_indices]
    Y_train, Y_valid = [Y[i] for i in train_indices], [Y[i] for i in valid_indices]

    print(f'Number of images: {len(X)}')
    print(f'- training: {len(train_indices)}')
    print(f'- validation: {len(valid_indices)}')
    return X_train, Y_train, X_valid, Y_valid
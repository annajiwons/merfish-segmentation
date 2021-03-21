import sys
sys.path.append('../')

import cv2 as cv
import numpy as np

from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

def get_confusion_matrix_vals(true, pred, thresholds):
    """
    TODO: write doc, assumes true and pred have same labels, returns dict with {thresh: (tp, fp, fn)} 
    """
    x = true.copy().ravel()
    y = pred.copy().ravel()

    # Get overlapping counts between x and y
    m, n = 1 + x.max(), 1 + y.max()
    overlap = np.zeros((m, n), dtype=np.uint)
    np.add.at(overlap.ravel(), x * n + y, 1)
    
    # Get matrix of IoU values for each pair of labels
    if np.sum(overlap) != 0:
        n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
        n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
        scores = overlap / (n_pixels_pred + n_pixels_true - overlap)
    else:
        scores = overlap
    
    scores = scores[1:,1:]
    n_true, n_pred = scores.shape
    n_matched = min(n_true, n_pred)

    cm_vals = {}
    
    for thresh in thresholds:
        # Get best pair matches and print IoU scores for each
        if n_matched > 0 and np.any(scores >= thresh):
            costs = -(scores >= thresh).astype(float) - scores / (2 * n_matched)
            true_ind, pred_ind = linear_sum_assignment(costs)
            match_ok = scores[true_ind, pred_ind] >= thresh
            tp = np.count_nonzero(match_ok)
        else:
            tp = 0
        
        fp = n_pred - tp
        fn = n_true - tp

        cm_vals[thresh] = (tp, fp, fn)

    return cm_vals

def watershed_seg(img, upper_limit, alpha, beta, blur_ksize, thresh_method, morph_kernel, opening_itr, dilate_itr, foreground_thresh_ratio, zero_boundaries):

    # Convert to 8bit grayscale
    img_gray = cv.cvtColor(np.copy(img), cv.COLOR_BGR2GRAY)

    # Remove pixels that are above the upper_limit
    img_gray[img_gray > upper_limit] = 0

    # Adjust brightness and contrast
    img_gray = cv.convertScaleAbs(img_gray, alpha = alpha, beta = beta)

    # Median filtering to remove noise
    img_gray_filter = cv.medianBlur(img_gray, blur_ksize)

    # Threshold foreground and background using Triangle thresholding
    thresh_val, thresh_img = cv.threshold(img_gray_filter, 0, 255, cv.THRESH_TRIANGLE)

    # Remove noise further using Opening morphological operation
    kernel = np.ones((3,3), np.uint8) if morph_kernel is None else morph_kernel
    opening = cv.morphologyEx(thresh_img, cv.MORPH_OPEN, kernel, iterations = opening_itr)

    # Find area that is sure to be background using dilate
    background = cv.dilate(opening, kernel, iterations = dilate_itr)

    # Find area that is sure to be the foreground
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    _, foreground = cv.threshold(dist_transform, foreground_thresh_ratio * dist_transform.max(), 255, cv.THRESH_BINARY)

    # Find area that is left, which is unknown to be foreground or background
    foreground = np.uint8(foreground)
    unknown = cv.subtract(background, foreground)

    # Marker labelling
    _, markers = cv.connectedComponents(foreground)

    # Add 1 to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Watershed
    markers = cv.watershed(img, markers)

    # Restore label values by subtracting 1, and zero out boundaries if option selected
    markers -= 1
    if zero_boundaries:
        markers[markers == -2] = 0
    else:
        markers[markers == -2] = -1
    return markers

def evaluate_all(imgs, masks, params):
    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tp_total = {key: 0 for key in taus}
    fp_total = {key: 0 for key in taus}
    fn_total = {key: 0 for key in taus}

    upper_limit = params['upper_limit']
    alpha = params['alpha']
    beta = params['beta']
    blur_ksize = params['blur_ksize']
    thresh_method = params['thresh_method']
    morph_kernel = params['morph_kernel']
    opening_itr = params['opening_itr']
    dilate_itr = params['dilate_itr']
    foreground_thresh_ratio = params['foreground_thresh_ratio']

    for i in range(len(imgs)):
        res = watershed_seg(imgs[i], upper_limit = upper_limit, alpha = alpha, beta = beta, blur_ksize = blur_ksize, 
                            thresh_method = thresh_method, morph_kernel = morph_kernel, opening_itr = opening_itr, 
                            dilate_itr = dilate_itr, foreground_thresh_ratio = foreground_thresh_ratio, zero_boundaries = True)

        cm_vals = get_confusion_matrix_vals(masks[i], res, taus)

        for tau in taus:
            tp_total[tau] += cm_vals[tau][0]
            fp_total[tau] += cm_vals[tau][1]
            fn_total[tau] += cm_vals[tau][2]
    return tp_total, fp_total, fn_total

def get_best_params(imgs, masks, param_tries):
    grid_options = {'upper_limit': np.random.randint(230, 257, size = 5),
                  'alpha': [0.5, 1, 1.5, 2],
                  'beta': [0, 5, 10, 15, 20],
                  'blur_ksize': [3, 5, 7, 9, 11], 
                  'thresh_method': [cv.THRESH_TRIANGLE, cv.THRESH_OTSU],
                  'morph_kernel': [np.ones((2,2), np.uint8), np.ones((3,3), np.uint8), np.ones((4,4), np.uint8), np.ones((5,5), np.uint8), np.ones((6,6), np.uint8)],
                  'opening_itr': [1, 2, 3, 4, 5],
                  'dilate_itr': [1, 2, 3, 4, 5],
                  'foreground_thresh_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    grid = ParameterGrid(grid_options)
    
    param_choices = np.random.choice(grid, param_tries)
    param_scores = []
    for params in tqdm(param_choices):
        tp_total, fp_total, fn_total = evaluate_all(imgs, masks, params)
        param_scores.append((params, tp_total, fp_total, fn_total))
    
    max_tp = 0
    for i in range(len(param_scores)):
        if param_scores[i][1][0.7] > max_tp:
            max_tp = param_scores[i][1][0.7]
            max_tp_idx = i
    
    return param_scores[max_tp_idx]

def f1_score(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0

def avg_precision(tp, fp, fn):
    return tp / (tp + fn + fp) if tp > 0 else 0
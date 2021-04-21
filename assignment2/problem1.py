import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def load_data(path):
    '''
    Load data from folder data, face images are in the folder facial_images, face features are in the folder facial_features.
    

    Args:
        path: path of folder data

    Returns:
        imgs: list of face images as numpy arrays 
        feats: list of facial features as numpy arrays 
    '''

    imgs = []
    feats = []

    # under the files of path, read all images
    path_im = path + '/facial_images/'
    path_fe = path + '/facial_features/'

    imgs_path = [path_im + x for x in os.listdir(path_im)]
    feats_path = [path_fe + x for x in os.listdir(path_fe)]

    imgs = [[] for i in range(len(imgs_path))]
    feats = [[] for i in range(len(feats_path))]

    for i in range(len(imgs_path)):
        im = plt.imread(imgs_path[i])
        imgs[i] = im

    for i in range(len(feats_path)):
        fe = plt.imread(feats_path[i])
        feats[i] = fe

    return imgs, feats

def gaussian_kernel(fsize, sigma):
    '''
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: sigma of Gaussian kernel

    Returns:
        The Gaussian kernel
    '''

    #
    kernel = np.empty((fsize, fsize))
    center = fsize // 2  # definition of the Gaussian kernel center

    for i in range(fsize):
        for j in range(fsize):
            x, y = i - center, j - center  # all pixels subtract the center
            kernel[i, j] = (1/(2*np.pi*sigma**2))*np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    return kernel

def downsample_x2(x, factor=2):
    '''
    Downsampling an image by a factor of 2

    Args:
        x: image as numpy array (H * W)

    Returns:
        downsampled image as numpy array (H/2 * W/2)
    '''

    #
    w, h = x.shape
    w_new, h_new = int(w / factor), int(h / factor)

    # only retain the rows with odd number
    downsample = np.empty((w_new, h_new))
    for i in range(w_new):
        for j in range(h_new):
            downsample[i, j] = x[2*i+1, 2*j+1]

    return downsample


def gaussian_pyramid(img, nlevels, fsize, sigma):
    '''
    A Gaussian pyramid is constructed by combining a Gaussian kernel and downsampling.
    Tips: use scipy.signal.convolve2d for filtering image.

    Args:
        img: face image as numpy array (H * W)
        nlevels: number of levels of Gaussian pyramid, in this assignment we will use 3 levels
        fsize: Gaussian kernel size, in this assignment we will define 5
        sigma: sigma of Gaussian kernel, in this assignment we will define 1.4

    Returns:
        GP: list of Gaussian downsampled images, it should be 3 * H * W
    '''
    GP = []

    #
    GP = [[] for k in range(nlevels)]
    w, h = img.shape
    kernel = gaussian_kernel(fsize, sigma)
    img_con = convolve2d(img, kernel, boundary='symm', mode='same')  # the convolution of image and kernel

    for level in range(nlevels):  # the number of pyramid levels
        w_new, h_new = int(w / (2 ** level)), int(h / (2 ** level))  # the size of every level reduces
        img_pyr = np.empty((w_new, h_new))
        for i in range(w_new):
            for j in range(h_new):  # after convolution is the down-sampling
                img_pyr[i, j] = img_con[(2 ** level) * i, (2 ** level) * j]
        GP[level].append(img_pyr)

    GP = [GP[i][0] for i in range(len(GP))]

    return GP


def template_distance(v1, v2):
    '''
    Calculates the distance between the two vectors to find a match.
    Browse the course slides for distance measurement methods to implement this function.
    Tips: 
        - Before doing this, let's take a look at the multiple choice questions that follow. 
        - You may need to implement these distance measurement methods to compare which is better.

    Args:
        v1: vector 1
        v2: vector 2

    Returns:
        Distance
    '''
    # choose the method SSD
    distance = 0
    for i in range(len(v1)):
        distance += (v1[i] - v2[i]) ** 2

    return distance


def sliding_window(img, feat, step=1):
    ''' 
    A sliding window for matching features to windows with SSDs. When a match is found it returns to its location.
    
    Args:
        img: face image as numpy array (H * W)
        feat: facial feature as numpy array (H * W)
        step: stride size to move the window, default is 1
    Returns:
        min_score: distance between feat and window
    '''

    img = img / 255
    feat = feat / 255  # normalization of all images
    distance = 0

    if len(img) < len(feat):  # perhaps the img size is smaller than feature size, so padding processing
        # want to expand the edge information
        img = np.pad(img, ((len(img), len(feat)), (len(img[0]), len(feat[0]))), 'symmetric')
    # SSD score
    min_score = np.empty(((len(img) // step) - len(feat)+1, (len(img[0]) // step) - len(feat[0])+1))

    for i in range((len(img) // step) - len(feat)+1):
        for j in range((len(img[0]) // step) - len(feat[0])+1):
            for k in range(len(feat)):
                for l in range(len(feat[0])):
                    distance += (img[i + k][j + l] - feat[k][l]) ** 2
            min_score[i, j] = distance
    min_score = min_score.reshape(-1)

    return min(min_score)


class Distance(object):

    # choice of the method
    METHODS = {1: 'Dot Product', 2: 'SSD Matching'}

    # choice of reasoning
    REASONING = {
        1: 'it is more computationally efficient',
        2: 'it is less sensitive to changes in brightness.',
        3: 'it is more robust to additive Gaussian noise',
        4: 'it can be implemented with convolution',
        5: 'All of the above are correct.'
    }

    def answer(self):
        '''Provide your answer in the return value.
        This function returns one tuple:
            - the first integer is the choice of the method you will use in your implementation of distance.
            - the following integers provide the reasoning for your choice.
        Note that you have to implement your choice in function template_distance

        For example (made up):
            (1, 1) means
            'I will use Dot Product because it is more computationally efficient.'
        '''

        return (2, 5)


def find_matching_with_scale(imgs, feats):
    ''' 
    Find face images and facial features that match the scales 
    
    Args:
        imgs: list of face images as numpy arrays
        feats: list of facial features as numpy arrays 
    Returns:
        match: all the found face images and facial features that match the scales: N * (score, g_im, feat)
        score: minimum score between face image and facial feature
        g_im: face image with corresponding scale
        feat: facial feature
    '''
    match = []
    (score, g_im, feat) = (None, None, None)

    # our method: every feature convolutes all images(all levels)
    match = [[] for l in range(len(feats))]
    gp_min = [[] for l in range(len(imgs))]
    img_min = [[] for l in range(len(feats))]
    score_gp = np.empty((len(imgs)))
    score_img = np.empty((len(feats)))
    for i in range(len(feats)):  # calculation of every feature
        feat_o = feats[i]
        for j in range(len(imgs)):  # calculation the score in every image
            img_gps = gaussian_pyramid(imgs[j], 3, 5, 1.4)
            score_gps = np.empty(len(img_gps))
            for k in range(len(img_gps)):  # calculation the score in every Gaussian pyramid level
                score_level = sliding_window(img_gps[k], feats[i], step=1)
                score_gps[k] = score_level
            score_gps = [x for x in score_gps]
            score_gp[j] = min(score_gps)  # the minimal score of all levels in one image
            gp_min[j] = img_gps[score_gps.index(min(score_gps))]
        score_gp = [y for y in score_gp]
        score_img[i] = min(score_gp)  # the minimal score of all images
        img_min[i] = gp_min[score_gp.index(min(score_gp))]  # which image(total 15 images) with minimal distance
        match[i] = (score_img[i], img_min[i], feat_o)

    return match
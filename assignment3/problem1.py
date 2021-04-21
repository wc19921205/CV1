import os
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

#
# Hint: you can make use of this function
# to create Gaussian kernels for different sigmas

"""
def gaussian_kernel(fsize=7, sigma=1):
    _x = _y = (fsize - 1) / 2
    x, y = np.mgrid[-_x:_x + 1, -_y:_y + 1]
    G = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    return G / (2 * np.pi * sigma**2)"""

# create a normalized gaussian function

def gaussian_kernel(fsize, sigma):
    '''
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: sigma of Gaussian kernel

    Returns:
        The Gaussian kernel
    '''

    kernel = np.empty((fsize, fsize))
    center = fsize // 2  # definition of the Gaussian kernel center

    for i in range(fsize):
        for j in range(fsize):
            x, y = i - center, j - center  # all pixels subtract the center
            kernel[i, j] = (1/(2*np.pi*sigma**2))*np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    return kernel


def normalization(k):
    '''
     normalization of the kernel in interval [-1, 1]

     Args:
         Gaussian kernel

     Returns:
         The normalized Gaussian kernel
     '''

    kern = k.reshape(-1)
    kernel_mean = np.mean(kern)
    kernel = (k - kernel_mean) / (max(kern) - min(kern))

    return kernel


def load_image(path):
    ''' 
    The input image is a) loaded, b) converted to greyscale, and
     c) converted to numpy array [-1, 1].

    Args:
        path: the name of the inpit image
    Returns:
        img: numpy array containing image in greyscale
    '''
    # load images and convert to greyscale
    img = np.array(Image.open(path).convert('L'), 'f')

    # normalization the image in interval [-1, 1]
    image = img.reshape(-1)
    img_max = max(image)
    img_min = min(image)
    img_mean = np.mean(image)

    img = (img - img_mean) / (img_max - img_min)

    return img

def smoothed_laplacian(image, sigmas, lap_kernel):
    ''' 
    Then laplacian operator is applied to the image and
     smoothed by gaussian kernels for each sigma in the list of sigmas.


    Args:
        Image: input image
        sigmas: sigmas specifying the scale
    Returns:
        response: 3 dimensional numpy array. The first (index 0) dimension is for scale
                  corresponding to sigmas
    '''

    img_smooth = np.empty((len(sigmas), image.shape[0], image.shape[1]))

    for i in range(len(sigmas)):
        g_kern = gaussian_kernel(7, sigmas[i])  # create a gaussian kernel, 7*7 kernel
        g_kern = normalization(g_kern)  # normalization of gaussian kernel
        img_smooth[i, :, :] = convolve2d(image, g_kern, boundary='symm', mode='same')
        img_smooth[i, :, :] = convolve2d(img_smooth[i], lap_kernel, boundary='symm', mode='same')

    return img_smooth

def laplacian_of_gaussian(image, sigmas):
    ''' 
    Then laplacian of gaussian operator for every sigma in the list of sigmas is applied to the image.

    Args:
        Image: input image
        sigmas: sigmas specifying the scale
    Returns:
        response: 3 dimensional numpy array. The first (index 0) dimension is for scale
                  corresponding to sigmas
    '''

    img_LoG = np.empty((len(sigmas), image.shape[0], image.shape[1]))

    for i in range(len(sigmas)):
        kern = LoG_kernel(9, sigmas[i])  # 9*9 kernel
        img_LoG[i, :, :] = convolve2d(image, kern, boundary='symm', mode='same')

    return img_LoG


def difference_of_gaussian(image, sigmas):
    ''' 
    Then difference of gaussian operator for every sigma in the list of sigmas is applied to the image.

    Args:
        Image: input image
        sigmas: sigmas specifying the scale
    Returns:
        response: 3 dimensional numpy array. The first (index 0) dimension is for scale
                  corresponding to sigmas
    '''

    img_DoG = np.empty((len(sigmas), image.shape[0], image.shape[1]))

    for i in range(len(sigmas)):
        img_DoG[i, :, :] = convolve2d(image, DoG(sigmas[i]), boundary='symm', mode='same')

    return img_DoG

def LoG_kernel(fsize, sigma):
    '''
    Define a LoG kernel.
    Tip: First calculate the second derivative of a gaussian and then discretize it.
    Args:
        fsize: kernel size
        sigma: sigma of guassian kernel

    Returns:
        LoG kernel
    '''
    # Approximation [[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [1, 2, -16, 2, 1], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]]

    kernel = np.empty((fsize, fsize))
    center = fsize // 2  # definition of the Gaussian kernel center

    for i in range(fsize):
        for j in range(fsize):
            x, y = i - center, j - center
            # Analytical calculation of the derivative of Gaussian kern
            kernel[i, j] = (-1 / (2 * np.pi * sigma ** 4)) * (np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))) * (
                        2 - (x ** 2 + y ** 2) / sigma ** 2)

    LoG_kern = normalization(kernel)

    return LoG_kern

def blob_detector(response):
    '''
    Find unique extrema points (maximum or minimum) in the response using 9x9 spatial neighborhood 
    and across the complete scale dimension.
    Tip: Ignore the boundary windows to avoid the need of padding for simplicity.
    Tip 2: unique here means skipping an extrema point if there is another point in the local window
            with the same value
    Args:
        response: 3 dimensional response from LoG operator in scale space.

    Returns:
        list of 3-tuples (scale_index, row, column) containing the detected points.
    '''

    result = []
    img_ext = {}

    for i in range(0, response.shape[1] - 8):
        for j in range(0, response.shape[2] - 8):
            img_blob = response[:, i:i + 9, j:j + 9]  # 9*9 matrix, there are scale spaces with number "scale_index"
            idx = np.unravel_index(img_blob.argmax(), img_blob.shape)  # get the position of local maximum in 9*9 square
            idx = idx[0], idx[1] + i, idx[2] + j  # the coordinate of the local maximum
            img_ext[idx] = response[idx[0], idx[1], idx[2]]  # dictionary. Keys: position, values: local max

    img_values = [v for v in img_ext.values()]
    img_values = sorted(img_values)
    val_p = np.percentile(img_values, 99.9)  # get the 99.9% count, than larger than this count
    img_values = [v for v in img_values if v >= val_p]  # only the 0.1% maximal values remain

    for k in img_ext.keys():  # return the coordinates of those local maximal points
        for l in range(len(img_values)):
            if img_ext[k] == img_values[l]:
                result.append(k)

    return result

def DoG(sigma):
    '''
    Define a DoG kernel. Please, use 9x9 kernels.
    Tip: First calculate the two gaussian kernels and return their difference. This is an approximation for LoG.

    Args:
        sigma: sigma of guassian kernel

    Returns:
        DoG kernel
    '''

    sigma_1 = sigma * np.sqrt(2)
    sigma_2 = sigma / np.sqrt(2)
    gaussian_1 = gaussian_kernel(9, sigma_1)
    gaussian_2 = gaussian_kernel(9, sigma_2)
    DoG_kern = gaussian_1 - gaussian_2  # get the kernel
    DoG_kern = normalization(DoG_kern)  # normalization of the kernel also in interval [-1, 1]

    return DoG_kern

def laplacian_kernel():
    '''
    Define a 3x3 laplacian kernel.
    Tip1: I_xx + I_yy
    Tip2: There are two possible correct answers.
    Args:
        none

    Returns:
        laplacian kernel
    '''

    laplace_kern = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # laplace_kern = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    # use these two approximations lead to two different results!!!

    return laplace_kern


class Method(object):

    # select one or more options
    REASONING = {
        1: 'it is always more computationally efficient',
        2: 'it is always more precise.',
        3: 'it always has fewer singular points',
        4: 'it can be implemented with convolution',
        5: 'All of the above are incorrect.'
    }

    def answer(self):
        '''Provide answer in the return value.
        This function returns a tuple containing indices of the correct answer.
        '''

        return (2, 3)

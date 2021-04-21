import numpy as np
import scipy.signal as signal

def rgb2bayer(image):
    """Convert image to bayer pattern:
    [B G]
    [G R]

    Args:
        image: Input image as (H,W,3) numpy array

    Returns:
        bayer: numpy array (H,W,3) of the same type as image
        where each color channel retains only the respective 
        values given by the bayer pattern
    """
    assert image.ndim == 3 and image.shape[-1] == 3

    # otherwise, the function is in-place
    bayer = image.copy()  # copy the original image, don't change its type

    height = image.shape[0]
    width = image.shape[1]
    # sweep all original
    for i in range(0, height, 2):
        for j in range(0, width, 2):
            # r
            bayer[i][j][0] = 0
            if i+1 < height:
                bayer[i+1][j][0] = 0
            if j+1 < width:
                bayer[i][j+1][0] = 0
            
            # g
            bayer[i][j][1] = 0
            if i+1 < height and j+1 < width:
                bayer[i+1][j+1][1] = 0
            
            # b
            if i+1 < height:
                bayer[i+1][j][2] = 0
            if j+1 < width:
                bayer[i][j+1][2] = 0
            if i+1 < height and j+1 < width:
                bayer[i+1][j+1][2] = 0

    assert bayer.ndim == 3 and bayer.shape[-1] == 3
    return bayer

def nearest_up_x2(x):
    """Upsamples a 2D-array by a factor of 2 using nearest-neighbor interpolation.

    Args:
        x: 2D numpy array (H, W)

    Returns:
        y: 2D numpy array if size (2*H, 2*W)
    """
    assert x.ndim == 2
    h, w = x.shape

    y = np.empty((2*h, 2*w))

    for i in range(0, 2*h, 1):
        for j in range(0, 2*w, 2):
            if i % 2 == 0:
                y[i][j] = x[i//2][j//2]
            else:
                y[i][j] = y[i-1][j]
        for j in range(1, 2*w, 2):
            y[i][j] = y[i][j-1]
        
    assert y.ndim == 2 and \
            y.shape[0] == 2*x.shape[0] and \
            y.shape[1] == 2*x.shape[1]
    return y

def bayer2rgb(bayer):
    """Interpolates missing values in the bayer pattern.
    Note, green uses bilinear interpolation; red and blue nearest-neighbour.

    Args:
        bayer: 2D array (H,W,C) of the bayer pattern
    
    Returns:
        image: 2D array (H,W,C) with missing values interpolated
        green_K: 2D array (3, 3) of the interpolation kernel used for green channel
        redblue_K: 2D array (3, 3) using for interpolating red and blue channels
    """
    assert bayer.ndim == 3 and bayer.shape[-1] == 3

    image = bayer.copy()
    rb_k = np.empty((3, 3))
    g_k = np.empty((3, 3))

    rb_k = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])  # use this appropriate kernel
    g_k = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])

    image[:, :, 0] = signal.convolve2d(image[:, :, 0], rb_k, mode='same', boundary='wrap')
    image[:, :, 1] = signal.convolve2d(image[:, :, 1], g_k, mode='same', boundary='wrap')/2
    image[:, :, 2] = signal.convolve2d(image[:, :, 2], rb_k, mode='same', boundary='wrap')

    assert image.ndim == 3 and image.shape[-1] == 3 and \
                g_k.shape == (3, 3) and rb_k.shape == (3, 3)
    return image, g_k, rb_k


def scale_and_crop_x2(bayer):
    """Upscamples a 2D bayer pattern by factor 2 and takes the central crop.

    Args:
        bayer: 2D array (H, W) containing bayer pattern

    Returns:
        image_zoom: 2D array (H, W) corresponding to x2 zoomed and interpolated 
        one-channel image
    """
    assert bayer.ndim == 2

    cropped = bayer.copy()
    width = cropped.shape[1]
    height = cropped.shape[0]

    cropped = cropped[height//4: height//4 + height//2, width//4: width//4 + width//2]
    #print(cropped[0:8, 0:8])
    cropped = nearest_up_x2(cropped)
    #print(cropped[0:8, 0:8])

    assert cropped.ndim == 2
    return cropped

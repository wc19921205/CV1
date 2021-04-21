import numpy as np
from scipy.signal import convolve2d
from scipy.interpolate import griddata

######################
# Basic Lucas-Kanade #
######################


def compute_derivatives(im1, im2):
    """Compute dx, dy and dt derivatives.
    
    Args:
        im1: first image
        im2: second image
    
    Returns:
        Ix, Iy, It: derivatives of im1 w.r.t. x, y and t
    """
    assert im1.shape == im2.shape

    # Your code here

    dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    dy = np.array([[-1, -1, 1], [0, 0, 0], [1, 1, 1]])
    dt = 0.25 * np.array([[-1, -1], [-1, -1]])

    Ix = convolve2d(im1, dx, boundary='symm', mode='same')
    Iy = convolve2d(im1, dy, boundary='symm', mode='same')
    It = convolve2d(im2, -dt, boundary='symm', mode='same') + convolve2d(im1, dt, boundary='symm', mode='same')
    
    assert Ix.shape == im1.shape and \
           Iy.shape == im1.shape and \
           It.shape == im1.shape

    return Ix, Iy, It


def compute_motion(Ix, Iy, It, patch_size=15, aggregate='none', sigma=2):
    """Computes one iteration of optical flow estimation.
    
    Args:
        Ix, Iy, It: image derivatives w.r.t. x, y and t
        patch_size: specifies the side of the square region R in Eq. (1)
        aggregate: 0 or 1 specifying the region aggregation region
        sigma: if aggregate=='gaussian', use this sigma for the Gaussian kernel
    Returns:
        u: optical flow in x direction
        v: optical flow in y direction
    
    All outputs have the same dimensionality as the input
    """
    assert Ix.shape == Iy.shape and \
        Iy.shape == It.shape

    # Your code here

    u = np.empty((Ix.shape[0], Ix.shape[1]))
    v = np.empty((Ix.shape[0], Ix.shape[1]))

    # processing the boundary
    Ix_pad = np.pad(Ix, patch_size // 2, 'reflect')
    Iy_pad = np.pad(Iy, patch_size // 2, 'reflect')
    It_pad = np.pad(It, patch_size // 2, 'reflect')

    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            # according to the equation in slide p55 u = -inv(first_term) * second_term
            first_term = np.zeros((2, 2))
            second_term = np.zeros(2)

            # selecting a region, accumulate the results of all points
            I_x = Ix_pad[i: i + patch_size, j: j + patch_size]
            I_y = Iy_pad[i: i + patch_size, j: j + patch_size]
            I_t = It_pad[i: i + patch_size, j: j + patch_size]

            for i_r in range(I_x.shape[0]):
                for j_r in range(I_x.shape[1]):
                    I_xy = np.array([I_x[i_r, j_r], I_y[i_r, j_r]])
                    # first_term += I_xy.dot(I_xy.T)

                    # task 8, adding the weight for all points (patch_size * patch_size)
                    # with a gaussian kernel with same size
                    if aggregate == 'gaussian':
                        kernel = gaussian_kernel(patch_size, sigma)
                        first_term = first_term + np.outer(I_xy, I_xy)
                        second_term = second_term + I_t[i_r, j_r] * I_xy
                        first_term = first_term * kernel[i_r, j_r]
                        second_term = second_term * kernel[i_r, j_r]
                    else:
                        first_term = first_term + np.outer(I_xy, I_xy)
                        second_term = second_term + I_t[i_r, j_r] * I_xy

            if np.linalg.det(first_term) == 0:
                u[i, j] = 0
                v[i, j] = 0
            else:
                u_v = -np.linalg.inv(first_term).dot(second_term)
                u[i, j] = u_v[0]
                v[i, j] = u_v[1]
    
    assert u.shape == Ix.shape and v.shape == Ix.shape
    return u, v


def warp(im, u, v):
    """Warping of a given image using provided optical flow.
    
    Args:
        im: input image
        u, v: optical flow in x and y direction
    
    Returns:
        im_warp: warped image (of the same size as input image)
    """
    assert im.shape == u.shape and \
        u.shape == v.shape

    # Your code here

    # obtain the mesh coordinates and then addition of the displacement.
    nx, ny = im.shape[1], im.shape[0]  # 584 * 388
    x, y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
    x_u = x + u
    y_v = y + v

    # obtain the coordinates of points in image
    Y = np.arange(0, ny, 1)
    for i in range(1, nx):
        Y = np.append(Y, np.arange(0, ny, 1))  # [0 1 2...387]  totally 388*584 points
    X = np.zeros(ny, dtype=np.int16)
    for i in range(1, nx):
        X = np.append(X, np.zeros(ny, dtype=np.int16) + i)  # [0 1 2...583]  totally 388*584 points
    points = np.vstack((Y, X)).T


    # obtain the values corresponding to points coordinates
    values = im[Y, X]

    # nearest interpolation
    im_warp = griddata(points, values, (y_v, x_u), method='linear')  # using method='linear', return 'nan'
    im_warp[np.isnan(im_warp)] = 0

    assert im_warp.shape == im.shape
    return im_warp


def compute_cost(im1, im2):
    """Implementation of the cost minimised by Lucas-Kanade."""
    assert im1.shape == im2.shape

    # Your code here
    dis = im1 - im2
    d = sum(dis.reshape(-1) ** 2)

    assert isinstance(d, float)
    return d

####################
# Gaussian Pyramid #
####################

#
# this function implementation is intentionally provided
#


def gaussian_kernel(fsize, sigma):
    """
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: deviation of the Guassian

    Returns:
        kernel: (fsize, fsize) Gaussian (normalised) kernel
    """

    _x = _y = (fsize - 1) / 2
    x, y = np.mgrid[-_x:_x + 1, -_y:_y + 1]
    G = np.exp(-0.5 * (x**2 + y**2) / sigma**2)

    return G / G.sum()


def downsample_x2(x, fsize=5, sigma=1.4):
    """
    Downsampling an image by a factor of 2

    Args:
        x: image as numpy array (H x W)
        fsize and sigma: parameters for Guassian smoothing
                         to apply before the subsampling
    Returns:
        downsampled image as numpy array (H/2 x W/2)
    """
    # Your code here

    # smoothing by the gaussian kernel
    im = convolve2d(x, gaussian_kernel(fsize, sigma), boundary='symm', mode='same')

    # new image
    w, h = im.shape[1], im.shape[0]  # original image: 388 * 584
    w_new, h_new = int(w / 2), int(h / 2)
    im_new = np.empty((h_new, w_new))

    # down-sampling with factor = 2, only retain the odd columns
    for i in range(h_new):
        for j in range(w_new):
            im_new[i, j] = im[i*2+1, j*2+1]

    return im_new


def gaussian_pyramid(img, nlevels=3, fsize=5, sigma=1.4):
    '''
    A Gaussian pyramid is constructed by combining a Gaussian kernel and downsampling.
    Tips: use scipy.signal.convolve2d for filtering image.

    Args:
        img: face image as numpy array (H * W)
        nlevels: num of level Gaussian pyramid, in this assignment we will use 3 levels
        fsize: gaussian kernel size, in this assignment we will define 5
        sigma: sigma of guassian kernel, in this assignment we will define 1.4

    Returns:
        GP: list of gaussian downsampled images, it shoud be 3 * H * W
    '''
    # Your code here

    img_pyramid = [img]

    # producing a list of gaussian pyramid with n levels, including original image
    for i in range(0, nlevels-1):
        img = downsample_x2(img, fsize, sigma)
        img_pyramid.append(img)

    return img_pyramid

###############################
# Coarse-to-fine Lucas-Kanade #
###############################


def coarse_to_fine(im1, im2, n_iter=5, nlevels=3):
    """Implementation of coarse-to-fine strategy
    for optical flow estimation.
    
    Args:
        im1, im2: first and second image
        pyramid1, pyramid2: Gaussian pyramids corresponding to im1 and im2
        n_iter: number of refinement iterations
    
    Returns:
        u: OF in x direction
        v: OF in y direction
    """
    assert im1.shape == im2.shape

    # Your code here

    dx = np.zeros((im1.shape[0] // (nlevels - 1) ** 2, im1.shape[1] // (nlevels - 1) ** 2))
    dy = np.zeros((im1.shape[0] // (nlevels - 1) ** 2, im1.shape[1] // (nlevels - 1) ** 2))

    # gaussian pyramids
    pyramid1 = gaussian_pyramid(im1, nlevels)
    pyramid2 = gaussian_pyramid(im2, nlevels)
    pyramid1 = list(reversed(pyramid1))
    pyramid2 = list(reversed(pyramid2))

    # Iteration of the coarse-to-fine
    # calculation from the image with smallest resolution in pyramid
    # After every iteration the image will be down-sampled
    # accumulate the flow between iterations without pre-warping
    for j in range(n_iter):
        print(j)
        for i in range(nlevels):
            Ix, Iy, It = compute_derivatives(pyramid1[i], pyramid2[i])
            u, v = compute_motion(Ix, Iy, It)
            # py_2 = warp(pyramid2[i], -u, -v)
            # Ix, Iy, It = compute_derivatives(pyramid2[i], py_2)
            # u_, v_ = compute_motion(Ix, Iy, It)
            # dx = u + u_ + dx
            # dy = v + v_ + dy
            dx = u + dx
            dy = v + dy

            # after every level up-sampling of the motion
            if i < nlevels-1:
                dx = nearest_up_x2(dx) * 2
                dy = nearest_up_x2(dy) * 2
                # dx = bilinear_up_x2(dx) * 2
                # dy = bilinear_up_x2(dy) * 2

        # down-sampling(twice) after every iteration
        if j < n_iter - 1:
            dx = downsample_x2(downsample_x2(dx) / 2) / 2
            dy = downsample_x2(downsample_x2(dy) / 2) / 2
        else:
            assert dx.shape == im1.shape and dy.shape == im1.shape
            return dx, dy


def nearest_up_x2(x):
    """Up-sampling a 2D-array by a factor of 2 using nearest-neighbor interpolation.

    Args:
        x: 2D numpy array (H, W)

    Returns:
        y: 2D numpy array if size (2*H, 2*W)
    """
    assert x.ndim == 2
    h, w = x.shape

    y = np.empty((2 * h, 2 * w))

    for i in range(0, 2 * h, 1):
        for j in range(0, 2 * w, 2):
            if i % 2 == 0:
                y[i][j] = x[i // 2][j // 2]
            else:
                y[i][j] = y[i - 1][j]

    assert y.ndim == 2 and \
           y.shape[0] == 2 * x.shape[0] and \
           y.shape[1] == 2 * x.shape[1]

    return y


def get_bilinear_filter(filter_shape, upscale_factor):
    """get a bilinear_filter in order to
       up-sampling a 2D-array by a factor of 2 using nearest-neighbor interpolation.

        Args:
            filter_shape: 1D numpy array, [width, height] = [4, 4]
            upscale_factor: in this task =2

        Returns:
            bilinear: a kernel is 2D numpy array with size (4, 4)
    """

    # calculation of the kernel_size. filter_shape is [width, height]
    kernel_size = 2 * upscale_factor - upscale_factor % 2

    # centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            value = (1 - abs((x - centre_location) / upscale_factor)) * \
                    (1 - abs((y - centre_location) / upscale_factor))
            bilinear[x, y] = value

    return bilinear


def bilinear_up_x2(im):
    """Up-sampling a 2D-array by a factor of 2 using bi-linear interpolation.

    Args: im: 2D numpy array (H, W)

    Returns: im_new: 2D numpy array if size (2*H, 2*W)
    """

    upscale_factor = 2

    im_new = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            im_new[2 * i, 2 * j] = im[i, j]

    im_new = convolve2d(im_new, get_bilinear_filter([4, 4], upscale_factor), boundary='symm', mode='same')

    return im_new


###############################
#   Multiple-choice question  #
###############################
def task9_answer():
    """
    Which statements about optical flow estimation are true?
    Provide the corresponding indices in a tuple.

    1. For rectified image pairs, we can estimate optical flow 
       using disparity computation methods.
    2. Lucas-Kanade method allows to solve for large motions in a principled way
       (i.e. without compromise) by simply using a larger patch size.
    3. Coarse-to-fine Lucas-Kanade is robust (i.e. negligible difference in the 
       cost function) to the patch size, in contrast to the single-scale approach.
    4. Lucas-Kanade method implicitly incorporates smoothness constraints, i.e.
       that the flow vector of neighbouring pixels should be similar.
    5. Lucas-Kanade is not robust to brightness changes.

    """

    return (1, 3, 4, 5)
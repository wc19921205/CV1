import numpy as np

def cost_ssd(patch1, patch2):
    """Compute the Sum of Squared Pixel Differences (SSD):
    
    Args:
        patch1: input patch 1 as (m, m, 1) numpy array
        patch2: input patch 2 as (m, m, 1) numpy array
    
    Returns:
        cost_ssd: the calcuated SSD cost as a floating point value
    """

    ssd = (patch1 - patch2) ** 2
    ssd = ssd.reshape(-1)
    ssd_cost = sum(ssd)

    assert np.isscalar(ssd_cost)
    return ssd_cost


def cost_nc(patch1, patch2):
    """Compute the normalized correlation cost (NC):

    Args:
        patch1: input patch 1 as (m, m, 1) numpy array
        patch2: input patch 2 as (m, m, 1) numpy array

    Returns:
        cost_nc: the calcuated NC cost as a floating point value
    """
    patch1_mean = np.mean(patch1.reshape(-1))
    patch2_mean = np.mean(patch2.reshape(-1))

    patch1_mean = patch1_mean * np.ones(patch1.reshape(-1).shape)
    patch2_mean = patch2_mean * np.ones(patch2.reshape(-1).shape)

    patch1_vector = patch1.reshape(-1)
    patch2_vector = patch2.reshape(-1)

    dis_1 = (patch1_vector - patch1_mean) ** 2
    patch1_dis = np.sqrt(sum(dis_1))

    dis_2 = (patch2_vector - patch2_mean) ** 2
    patch2_dis = np.sqrt(sum(dis_2))

    p1_dot_p2 = (patch1_vector - patch1_mean).T.dot(patch2_vector - patch2_mean)
    p1_dot_p2 = sum(p1_dot_p2.reshape(-1))
    nc_cost = p1_dot_p2 / patch1_dis * patch2_dis

    assert np.isscalar(nc_cost)
    return nc_cost


def cost_function(patch1, patch2, alpha):
    """Compute the cost between two input window patches given the disparity:
    
    Args:
        patch1: input patch 1 as (m, m) numpy array
        patch2: input patch 2 as (m, m) numpy array
        input_disparity: input disparity as an integer value        
        alpha: the weighting parameter for the cost function
    Returns:
        cost_val: the calculated cost value as a floating point value
    """
    assert patch1.shape == patch2.shape

    cost_val = cost_ssd(patch1, patch2) / (patch1.shape[0] ** 2) + alpha * cost_nc(patch1, patch2)
    # cost_val = -1
    
    assert np.isscalar(cost_val)
    return cost_val


def pad_image(input_img, window_size, padding_mode='symmetric'):
    """Output the padded image
    
    Args:
        input_img: an input image as a numpy array
        window_size: the window size as a scalar value, odd number
        padding_mode: the type of padding scheme, among 'symmetric', 'reflect', or 'constant'
        
    Returns:
        padded_img: padded image as a numpy array of the same type as image
    """
    assert np.isscalar(window_size)
    assert window_size % 2 == 1

    padded_img = input_img.copy()

    pad_width = window_size // 2
    padded_img = np.pad(padded_img, ((pad_width, pad_width),
                                     (pad_width, pad_width)), padding_mode)

    return padded_img


def compute_disparity(padded_img_l, padded_img_r, max_disp, window_size, alpha):
    """Compute the disparity map by using the window-based matching:    

    Args:
        padded_img_l: The padded left-view input image as 2-dimensional (H,W) numpy array
        padded_img_r: The padded right-view input image as 2-dimensional (H,W) numpy array
        max_disp: the maximum disparity as a search range
        window_size: the patch size for window-based matching, odd number
        alpha: the weighting parameter for the cost function
    Returns:
        disparity: numpy array (H,W) of the same type as image
    """

    assert padded_img_l.ndim == 2
    assert padded_img_r.ndim == 2
    assert padded_img_l.shape == padded_img_r.shape
    assert max_disp > 0
    assert window_size % 2 == 1

    #
    # Your code goes here
    #
    '''
    disparity = np.empty((padded_img_l.shape[0] - window_size, padded_img_l.shape[1] - window_size))

    for i in range(window_size, padded_img_l.shape[0] - window_size):
        for j in range(window_size + max_disp, padded_img_l.shape[1] - window_size):
            patch1 = padded_img_l[i:i + window_size, j:j + window_size]
            cost = []
            for k in range(0, max_disp):
                patch2 = padded_img_r[i:i + window_size, j-k:j + window_size - k]
                cost.append(cost_function(patch1, patch2, alpha) )
            #print("***", cost)
            disparity[i][j] = min(cost)
   # disparity = pad_image(disparity, window_size, padding_mode='symmetric')
    print(disparity)
    '''
    half_size = window_size // 2
    h1, w1 = padded_img_l.shape
    h2, w2 = padded_img_r.shape

    for i in range(h1 - window_size + 1):
        minr = i - half_size
        maxr = i + half_size

        for j in range(w1 - window_size + 1):
            minc = j - half_size
            maxc = j + half_size

            template = padded_img_r[minr:maxr, minc:maxc]

            disparity = np.empty((h2 - half_size, w2 - half_size))
            cost = []
            for k in range(max_disp + 1):
                block = padded_img_l[minr:maxr, (minc + k):(maxc + k)]
                cost.append(cost_function(block, template, alpha))
            cost_min = min(cost)
            disparity[i, j] = cost_min

    assert disparity.ndim == 2
    return disparity


def compute_aepe(disparity_gt, disparity_res):
    """Compute the average end-point error of the estimated disparity map:
    
    Args:
        disparity_gt: the ground truth of disparity map as (H, W) numpy array
        disparity_res: the estimated disparity map as (H, W) numpy array
    
    Returns:
        aepe: the average end-point error as a floating point value
    """
    assert disparity_gt.ndim == 2 
    assert disparity_res.ndim == 2 
    assert disparity_gt.shape == disparity_res.shape

    disparity_gt = disparity_gt.reshape(-1)
    disparity_res = disparity_res.reshape(-1)
    aepe = []

    for i in range(len(disparity_gt)):
        aepe.append(disparity_gt[i] - disparity_res[i])
    aepe = sum(aepe) / len(disparity_gt)
    # aepe = -1

    assert np.isscalar(aepe)
    return aepe

def optimal_alpha():
    """Return alpha that leads to the smallest EPE 
    (w.r.t. other values)"""
    
    #
    # Fix alpha
    #
    alpha = np.random.choice([-0.06, -0.01, 0.04, 0.1])


    return alpha


"""
This is a multiple-choice question
"""
class WindowBasedDisparityMatching(object):

    def answer(self):
        """Complete the following sentence by choosing the most appropriate answer 
        and return the value as a tuple.
        (Element with index 0 corresponds to Q1, with index 1 to Q2 and so on.)
        
        Q1. [?] is better for estimating disparity values on sharp objects and object boundaries
          1: Using a smaller window size (e.g., 3x3)
          2: Using a bigger window size (e.g., 11x11)
        
        Q2. [?] is good for estimating disparity values on locally non-textured area.
          1: Using a smaller window size (e.g., 3x3)
          2: Using a bigger window size (e.g., 11x11)

        Q3. When using a [?] padding scheme, the artifacts on the right border of the estimated disparity map become the worst.
          1: constant
          2: reflect
          3: symmetric

        Q4. The inaccurate disparity estimation on the left image border happens due to [?].
          1: the inappropriate padding scheme
          2: the absence of corresponding pixels
          3: the limitations of the fixed window size
          4: the lack of global information

        """

        return (-1, -1, 1, -1)

import numpy as np
import matplotlib.pyplot as plt
import copy

def display_image(img):
    """ Show an image with matplotlib:

    Args:
        Image as numpy array (H,W,3)
    """
    #
    # You code here
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def save_as_npy(path, img):
    """ Save the image array as a .npy file:

    Args:
        Image as numpy array (H,W,3)
    """
    
    #
    # You code here
    np.save(path, img)



def load_npy(path):
    """ Load and return the .npy file:

    Returns:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    return np.load(path)


def mirror_horizontal(img):
    """ Create and return a horizontally mirrored image:

    Args:
        Loaded image as numpy array (H,W,3)

    Returns:
        A horizontally mirrored numpy array (H,W,3).
    """
    
    #
    # You code here
    img_copy = copy.deepcopy(img)  # copy the original image, to avoid changing img1
    size = img_copy.shape
    h = size[0]  # height of image
    w = size[1]  # width of image

    for i in range(h):
        for j in range(w):
            img_copy[i, w-1-j] = img_copy[i, j]  # loop: the half of image(array) is equal to the another half part
    return img_copy


def display_images(img1, img2):
    """ display the normal and the mirrored image in one plot:

    Args:
        Two image numpy arrays
    """

    #
    # You code here
    figure = plt.figure(figsize=(20, 10))  # set the size of figure
    plt.axis("off")
    ax = figure.add_subplot(121)  # plot images with 1 row and 2 columns
    plt.axis('off')
    ax.imshow(img1)  # load the first image
    ax.set_title('image_1')

    ax = figure.add_subplot(122)
    plt.axis('off')
    ax.imshow(img2)  # load the second image
    ax.set_title('image_2')

    plt.show()


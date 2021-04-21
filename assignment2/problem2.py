import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

#
# Task 1
#
def load_faces(path, ext=".pgm"):
    """Load faces into an array (N, M),
    where N is the number of face images and
    d is the dimensionality (height*width for greyscale).
    
    Hint: os.walk() supports recursive listing of files 
    and directories in a path
    
    Args:
        path: path to the directory with face images
        ext: extension of the image files (you can assume .pgm only)
    
    Returns:
        x: (N, M) array
        hw: tuple with two elements (height, width)
    """
    
    # read the images
    filenames = [os.path.join(root, name) for root, dirs, files in os.walk(path, topdown=False)
                 for name in files if name.endswith(ext)]

    imgs = [[] for i in range(len(filenames))]
    for i in range(len(filenames)):
        img = plt.imread(filenames[i])
        img = img.T
        imgs[i] = img
        h, w = img.shape

    x = np.array(imgs)  # return all 760 original images, every image with size 84*96 (h*w)

    return x, (h, w)

#
# Task 2
#

"""
This is a multiple-choice question
"""

class PCA(object):

    # choice of the method
    METHODS = {
                1: "SVD",
                2: "Eigendecomposition"
    }

    # choice of reasoning
    REASONING = {
                1: "it can be applied to any matrix and is more numerically stable",
                2: "it is more computationally efficient for our problem",
                3: "it allows to compute eigenvectors and eigenvalues of any matrix",
                4: "we can find the eigenvalues we need for our problem from the singular values",
                5: "we can find the singular values we need for our problem from the eigenvalues"
    }

    def answer(self):
        """Provide answer in the return value.
        This function returns one tuple:
            - the first integer is the choice of the method you will use in your implementation of PCA
            - the following integers provide the reasoning for your choice

        For example (made up):
            (2, 1, 5) means
            "I will use eigendecomposition because
                - we can apply it to any matrix
                - we need singular values which we can obtain from the eigenvalues"
        """

        return (1, 3, 4)

# Task 3
#
mu = 0
std = 0


def compute_pca(X):
    """PCA implementation
    
    Args:
        X: (N, M) an array with N M-dimensional features
    
    Returns:
        u: (M, N) bases with principal components
        lmb: (N, ) corresponding variance
    """
    X_norm = np.empty((X.shape[0], X.shape[1], X.shape[2]))
    cov = np.empty((X.shape[0], X.shape[1], X.shape[1]))
    lmb = np.empty((X.shape[0], X.shape[1]))
    u = np.empty((X.shape[0], X.shape[1], X.shape[1]))

    y = X.reshape((-1, X.shape[2]))
    # print(y.shape)
    # print(X.shape)
    global mu
    global std

    # Normalization of images, mu and std will be used in other functions, so global declaration.
    mu = np.mean(y, axis=0)
    std = np.std(y, axis=0)

    for i in range(len(X)):
        # Matrix Normalization
        X_norm[i, :, :] = (X[i] - mu) / std
        # Calculation the covariance matrix, its eigenvalues and -vectors
        cov[i, :, :] = X_norm[i].dot(X_norm[i].T) / (X_norm[i].shape[0])
        v, e = np.linalg.eig(cov[i, :, :])
        lmb[i, :] = v
        u[i, :, :] = e
    
    return u, lmb

#
# Task 4
#

def basis(u, s, p = 0.5):
    """Return the minimum number of basis vectors 
    from matrix U such that they account for at least p percent
    of total variance.
    
    Hint: Do the singular values really represent the variance?
    
    Args:
        u: (M, M) contains principal components.
        For example, i-th vector is u[:, i]
        s: (M, ) variance along the principal components.
    
    Returns:
        v: (M, D) contains M principal components from N
        containing at most p (percentile) of the variance.
    
    """
    u_rank = np.empty((u.shape[0], u.shape[1], u.shape[1]))
    s_rank = np.empty((u.shape[0], u.shape[1]))
    u_low = []

    cumulative_s = np.empty((u.shape[0], u.shape[1]))
    # the sequence of eigenvalues after np.linalg.eig() is chaotic
    # sorting the eigenvalues and corresponding vectors
    for i in range(len(s)):
        idx = np.argsort(s[i])[::-1]
        u_rank[i, :, :] = u[i, :, idx]  # eigenvectors: according to the descending order
        s_rank[i, :] = s[i, idx]  # eigenvalues: according to the descending order
        cumulative_s[i, :] = np.cumsum(s_rank[i]) / np.sum(s_rank[i])  # the cumulative value of eigenvalues
        k = 0
        # only keep the first "p" percentage eigenvalues
        # and corresponding vectors
        for percent in cumulative_s[i]:
            k += 1
            if percent >= p:
                break
        u_low.append(u_rank[i, :, :k])  # in lower-dimension

    u = u_low
    
    return u

#
# Task 5
#
def project(face_image, u):
    """Project face image to a number of principal
    components specified by num_components.
    
    Args:
        face_image: (N, ) vector (N=h*w) of the face
        u: (N,M) matrix containing M principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        image_out: (N, ) vector, projection of face_image on 
        principal components
    """
    img_out = np.empty((len(u), face_image.shape[0], face_image.shape[1]))
    face_image = (face_image - mu) / std

    # re-projection of the test_image into the principle vectors of all images
    for i in range(len(u)):
        img_out[i, :, :] = (face_image.T.dot(u[i]).dot(u[i].T)).T
        # return is an original image, i.e. re-projection the face_image into its own eigenvector
        img_out[i, :, :] = img_out[i, :, :] * std + mu
    
    return img_out[0]

#
# Task 6
#

"""
This is a multiple-choice question
"""
class NumberOfComponents(object):

    # choice of the method
    OBSERVATION = {
                1: "The more principal components we use, the sharper is the image",
                2: "The fewer principal components we use, the smaller is the re-projection error",
                3: "The first principal components mostly correspond to local features, e.g. nose, mouth, eyes",
                4: "The first principal components predominantly contain global structure, e.g. complete face",
                5: "The variations in the last principal components are perceptually insignificant; these bases can be neglected in the projection"
    }

    def answer(self):
        """Provide answer in the return value.
        This function returns one tuple describing you observations

        For example: (1, 3)
        """

        return (3, 5)


#
# Task 7
#
def search(Y, x, u, top_n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        Y: (N, M) centered array with N d-dimensional features
        x: (1, M) image we would like to retrieve
        u: (M, D) basis vectors. Note, we already assume D has been selected.
        top_n: integer, return top_n closest images in L2 sense.
    
    Returns:
        Y: (top_n, M)
    """
    img_project = []
    test_project = []
    distance = []

    for i in range(Y.shape[0]):
        img_project.append(u[i].T.dot(Y[i]))  # projection of all images onto corresponding eigenvectors
        test_project.append(u[i].T.dot(x))  # projection of test image onto corresponding eigenvectors
        temp = np.linalg.norm(img_project[i] - test_project[i])  # calculation of norm to describe distance
        distance.append(temp)
    idx = np.argsort(distance)
    # print(idx.shape)
    idx = idx[:top_n]
    Y = Y[idx, :, :]  # return the original images

    return Y

#
# Task 8
#
def interpolate(x1, x2, u, N):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        x1: (1, M) array, the first image
        x2: (1, M) array, the second image
        u: (M, D) basis vectors. Note, we already assume D has been selected.
        N: number of interpolation steps (including x1 and x2)

    Hint: you can use np.linspace to generate N equally-spaced points on a line
    
    Returns:
        Y: (N, M) interpolated results. The first dimension is in the index into corresponding
        image; Y[0] == project(x1, u); Y[-1] == project(x2, u)
    """

    res_1 = []
    res_2 = []
    distance_1 = []
    distance_2 = []
    m = N + 2
    x1 = (x1 - mu) / std  # normalization
    x2 = (x2 - mu) / std
    # print(x1.shape)
    # print(x2.shape)

    # to find which u is the most approximate for the projection
    # i.e. want to find which u[i] the images x1, x2 belong to
    for i in range(len(u)):
        res_1.append((x1.T.dot(u[i]).dot(u[i].T)).T)
        res_2.append((x2.T.dot(u[i]).dot(u[i].T)).T)
        temp_1 = np.linalg.norm(res_1[i] - x1)  # through method calculation of norm
        temp_2 = np.linalg.norm(res_2[i] - x2)
        distance_1.append(temp_1)
        distance_2.append(temp_2)

    idx_1 = np.argsort(distance_1)[0]
    idx_2 = np.argsort(distance_2)[0]

    u_1 = u[idx_1]
    u_2 = u[idx_2]

    # Projection in lower-dimension
    img_low1 = x1.T.dot(u_1).dot(u_1.T)
    img_low2 = x2.T.dot(u_2).dot(u_2.T)
    img_interpol = np.empty((img_low1.shape[0], img_low1.shape[1], m))

    # Interpolation between two images (in low-dimension)
    for i in range(img_low1.shape[0]):
        for j in range(img_low1.shape[1]):
            if img_low1[i][j] < img_low2[i][j]:
                img_interpol[i, j, :] = np.linspace(img_low1[i][j], img_low2[i][j], num=m)
            else:
                img_interpol[i, j, :] = np.linspace(img_low2[i][j], img_low1[i][j], num=m)

    img_interpol = img_interpol.T

    # return the project image (*std + mu)
    img_orig = np.empty((img_interpol.shape[0], img_interpol.shape[1], img_interpol.shape[2]))
    for i in range(img_interpol.shape[0]):
        img_orig[i, :, :] = img_interpol[i] * std + mu
    
    return img_orig

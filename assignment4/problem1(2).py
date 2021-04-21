import numpy as np


def transform(pts):
    """Point conditioning: scale and shift points into [-1, 1] range
    as suggested in the lecture.
    
    Args:
        pts: [Nx2] numpy array of pixel point coordinates
    
    Returns:
        T: [3x3] numpy array, such that Tx normalises 
            2D points x (in homogeneous form).
    
    """
    assert pts.ndim == 2 and pts.shape[1] == 2

    # create the transformation matrix for normalization
    s_norm = np.empty((pts.shape[0]))
    for i in range(pts.shape[0]):
        s_norm[i] = np.linalg.norm(pts[i])  # norm
    s = 1/2 * max(s_norm)
    t = np.mean(pts, axis=0)  # mean
    T = np.array([[1 / s, 0, -t[0] / s], [0, 1 / s, -t[1] / s], [0, 0, 1]])

    assert T.shape == (3, 3)
    return T


def transform_pts(pts, T):
    """Applies transformation T on 2D points.
    
    Args:
        pts: (Nx2) 2D point coordinates
        T: 3x3 transformation matrix
    
    Returns:
        pts_out: (Nx3) transformed points in homogeneous form
    
    """
    assert pts.ndim == 2 and pts.shape[1] == 2
    assert T.shape == (3, 3)

    # normalization of the corresponding points
    matrix = T
    pts_new = np.ones((pts.shape[0], 3))  # creat the homogeneous matrix
    pts_new[:, :2] = pts
    pts_trans = matrix.dot(pts_new.T).T
    
    assert pts_trans.shape == (pts.shape[0], 3)
    return pts_trans


def create_A(pts1, pts2):
    """Create matrix A such that our problem will be Ax = 0,
    where x is a vectorised representation of the 
    fundamental matrix.
        
    Args:
        pts1 and pts2: Nx2 numpy arrays corresponding to 2D points 
    
    Returns:
        A: numpy array
    """
    assert pts1.shape == pts2.shape

    t_1 = transform(pts1)
    t_2 = transform(pts2)

    p1 = transform_pts(pts1, t_1)
    p2 = transform_pts(pts2, t_2)

    p1 = p1 / p1[2]
    p2 = p2 / p2[2]

    p1 = p1[:, :2]
    p2 = p2[:, :2]

    # num = np.random.randint(0, pts1.shape[0], size=8)
    # p1 = p1[num, :]
    # p2 = p2[num, :]

    a_matrix = np.empty((p1.shape[0], 9))
    for i in range(a_matrix.shape[0]):
        a_matrix[i, :] = [p2[i][0] * p1[i][0], p2[i][1] * p1[i][0], p1[i][0], p2[i][0] * p1[i][1],
                          p2[i][1] * p1[i][1], p1[i][1], p2[i][0], p2[i][1], 1]

    return a_matrix


def enforce_rank2(F):
    """Enforce rank 2 of 3x3 matrix
    
    Args:
        F: 3x3 matrix
    
    Returns:
        F_out: 3x3 matrix with rank 2
    """
    assert F.shape == (3, 3)

    u, s, v = np.linalg.svd(F)
    d = np.array([[s[0], 0, 0], [0, s[1], 0], [0, 0, 0]])  # s[2] = 0
    F_final = u.dot(d).dot(v)  # matrix v is already transposed

    assert F_final.shape == (3, 3)
    return F_final


def compute_F(A):
    """Computing the fundamental matrix from F
    by solving homogeneous least-squares problem
    Ax = 0, subject to ||x|| = 1
    
    Args:
        A: matrix A
    
    Returns:
        f: 3x3 matrix subject to rank-2 contraint
    """

    u, s, v = np.linalg.svd(A, full_matrices=False)  # produce the u, s, v matrix
    F_final = v[-1]
    F_final = F_final.reshape((3, 3))  # the fundamental matrix: rank = 3

    F_final = enforce_rank2(F_final)  # reduce the rank to 2
    
    assert F_final.shape == (3, 3)
    return F_final


def compute_residual(F, x1, x2):
    """Computes the residual g as defined in the assignment sheet.
    
    Args:
        F: fundamental matrix
        x1,x2: point correspondences
    
    Returns:
        float
    """

    # create the homogeneous coordinates
    x1_new = np.ones((x1.shape[0], 3))
    x1_new[:, :2] = x1

    x2_new = np.ones((x2.shape[0], 3))
    x2_new[:, :2] = x2

    # calculation of the residual by the original (not normalized) F_matrix and points
    g = 0
    for i in range(x1_new.shape[0]):
        g = g + x1_new[i].T.dot(F).dot(x2_new[i])
    g = g / x1_new.shape[0]

    return g


def denorm(F, T1, T2):
    """Denormalising matrix F using 
    transformations T1 and T2 which we used
    to normalise point coordinates x1 and x2,
    respectively.
    
    Returns:
        3x3 denormalised matrix F
    """

    F_original = T2.T.dot(F).dot(T1)

    return F_original


def estimate_F(x1, x2, t_func):
    """Estimating fundamental matrix from pixel point
    coordinates x1 and x2 and normalisation specified 
    by function t_func (t_func returns 3x3 transformation 
    matrix).
    
    Args:
        x1, x2: 2D pixel coordinates of matching pairs
        t_func: normalising function (for example, transform)
    
    Returns:
        F: fundamental matrix
        res: residual g defined in the assignment
    """
    
    assert x1.shape[0] == x2.shape[0]

    a_matrix = create_A(x1, x2)
    f_matrix = compute_F(a_matrix)
    F = denorm(f_matrix, t_func(x1), t_func(x2))

    res = compute_residual(F, x1, x2)

    return F, res


def line_y(xs, F, pts):
    """Compute corresponding y coordinates for 
    each x following the epipolar line for
    every point in pts.
    
    Args:
        xs: N-array of x coordinates
        F: fundamental matrix (3*3)
        pts: (Mx3) array specifying pixel corrdinates
             in homogeneous form.
    
    Returns:
        MxN array containing y coordinates of epipolar lines.
    """
    N, M = xs.shape[0], pts.shape[0]
    assert F.shape == (3, 3)

    pts_trans = F.T.dot(pts.T).T  # M*3
    ys = np.empty((pts.shape[0], xs.shape[0]))  # M*3, N

    # according to the property of F and points in two views -> calculation of the y_line
    for i in range(pts.shape[0]):  # M*3
        for j in range(xs.shape[0]):  # N
            ys[i][j] = -(xs[j] * pts_trans[i][0] + pts_trans[i][2]) / pts_trans[i][1]

    assert ys.shape == (M, N)
    return ys


#
# Bonus tasks
#

import math


def transform_v2(pts):
    """Point conditioning: scale and shift points into [-1, 1] range.
    
    Args:
        pts1 and pts2: Nx2 numpy arrays corresponding to 2D points
    
    Returns:
        T: numpy array, such that Tx conditions 2D (homogeneous) points x.
    
    """

    t = np.mean(pts, axis=0)
    s = np.std(pts, axis=0)
    pts = (pts - t) / s  # centroid is zero

    dis_1 = 0
    dis_2 = 0
    for i in range(p1.shape[0]):
        dis_1 = dis_1 + np.sqrt(pts[i][0] ** 2)
        dis_2 = dis_2 + np.sqrt(pts[i][1] ** 2)
    dis_1 = distance_1 / pts.shape[0]
    dis_2 = distance_2 / pts.shape[0]

    T = np.array([[1 / (s[0] * dis_1), 0, -t[0] / (s[0] * dis_1)],
                  [0, 1 / (s[1] * dis_2), -t[1] / (s[1] * dis_2)], [0, 0, 1]])

    return T


"""Multiple-choice question"""
class MultiChoice(object):

    """ Which statements about fundamental matrix F estimation are true?

    1. We need at least 7 point correspondences to estimate matrix F.
    2. We need at least 8 point correspondences to estimate matrix F.
    3. More point correspondences will not improve accuracy of F as long as 
    the minimum number of points correspondences are provided.
    4. Fundamental matrix contains information about intrinsic camera parameters.
    5. One can recover the rotation and translation (up to scale) from the essential matrix 
    corresponding to the transform between the two views.
    6. The determinant of the fundamental matrix is always 1.
    7. Different normalisation schemes (e.g. transform, transform_v2) may have
    a significant effect on estimation of F. For example, epipoles can deviate.
    (Hint for 7): Try using corridor image pair.)

    Please, provide the indices of correct options in your answer.
    """

    def answer(self):
        return (1, 4, 5)


def compute_epipole(F, eps=1e-8):
    """Compute epipole for matrix F,
    such that Fe = 0.
    
    Args:
        F: fundamental matrix
    
    Returns:
        e: 2D vector of the epipole
    """
    assert F.shape == (3, 3)

    u, s, v = np.linalg.svd(F, full_matrices=False)
    e = v[-1]
    e = e / e[2]
    e = e[:2]
    # e = np.empty((2, ))

    return e

def intrinsics_K(f=1.05, h=480, w=640):
    """Return 3x3 camera matrix.
    
    Args:
        f: focal length (same for x and y)
        h, w: image height and width
    
    Returns:
        3x3 camera matrix
    """
    # K = np.empty((3, 3))

    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])

    return K

def compute_E(F):
    """Compute essential matrix from the provided
    fundamental matrix using the camera matrix (make 
    use of intrinsics_K).

    Args:
        F: 3x3 fundamental matrix

    Returns:
        E: 3x3 essential matrix
    """
    # E = np.empty((3, 3))
    K = intrinsics_K(1.05, 480, 640)
    E = K.T.dot(F).dot(K)

    return E
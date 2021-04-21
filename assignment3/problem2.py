import numpy as np
import matplotlib.pyplot as plt


def load_pts_features(path):
    """ Load interest points and SIFT features.

    Args:
        path: path to the file pts_feats.npz
    
    Returns:
        pts: coordinate points for two images;
             an array (2,) of numpy arrays (N1, 2), (N2, 2)
        feats: SIFT descriptors for two images;
               an array (2,) of numpy arrays (N1, 128), (N2, 128)
    """

    keys = np.load(path, allow_pickle=True)

    points_img1 = keys['pts'][0]
    points_img2 = keys['pts'][1]
    feats_img1 = keys['feats'][0]
    feats_img2 = keys['feats'][1]

    feats = [feats_img1, feats_img2]
    pts = [points_img1, points_img2]

    return pts, feats

def min_num_pairs():

    return 4

def pickup_samples(pts1, pts2):
    """ Randomly select k corresponding point pairs.
    Note that here we assume that pts1 and pts2 have 
    been already aligned: pts1[k] corresponds to pts2[k].

    This function makes use of min_num_pairs()

    Args:
        pts1 and pts2: point coordinates from Image 1 and Image 2
    
    Returns:
        pts1_sub and pts2_sub: N_min randomly selected points 
                               from pts1 and pts2
    """
    n = min_num_pairs()
    # random = np.random.RandomState(0)  # random seed
    # create the 4 points randomly
    num = np.random.randint(0, pts2.shape[0], size=n)

    pts1_sub = pts1[num, :]
    pts2_sub = pts2[num, :]

    return pts1_sub, pts2_sub


def compute_homography(pts1, pts2):
    """ Construct homography matrix and solve it by SVD

    Args:
        pts1: the coordinates of interest points in img1, array (N, 2)
        pts2: the coordinates of interest points in img2, array (M, 2)
    
    Returns:
        H: homography matrix as array (3, 3)
    """

    # get four-pairs corresponding points
    p_1, p_2 = pickup_samples(pts1, pts2)

    # write the A matrix because of Ah = 0
    matrix_A = np.empty((2 * p_1.shape[0], 9))
    for i in range(p_1.shape[0]):
        matrix_A[2*i, :] = [0, 0, 0, p_1[i][0], p_1[i][1], 1, -p_1[i][0]*p_2[i][1], -p_1[i][1]*p_2[i][1], -p_2[i][1]]
        matrix_A[2*i+1, :] = [-p_1[i][0], -p_1[i][1], -1, 0, 0, 0, p_1[i][0]*p_2[i][0], p_1[i][1]*p_2[i][0], p_2[i][0]]

    # calculate the SVD and only the minimal eigenvalue and corresponding vector remain
    u, s, v = np.linalg.svd(matrix_A, full_matrices=False)
    homo = v[-1]
    homo = homo.reshape((3, 3))

    return homo


def transform_pts(pts, H):
    """ Transform pst1 through the homography matrix to compare pts2 to find inliners

    Args:
        pts: interest points in img1, array (N, 2)
        H: homography matrix as array (3, 3)
    
    Returns:
        transformed points, array (N, 2)
    """

    # expanding the original matrix to calculate the corresponding point in another image
    pts_new = np.ones((pts.shape[0], 3))
    pts_new[:, :2] = pts  # create a homogeneous matrix

    pts_trans = np.dot(H, pts_new.T).T
    for i in range(pts_trans.shape[0]):
        pts_trans[i] = pts_trans[i] / pts_trans[i, 2]  # normalization
    pts_trans = pts_trans[:, :2]  # only two columns of each point

    return pts_trans


def count_inliers(H, pts1, pts2, threshold=5):
    """ Count inliers
        Tips: We provide the default threshold value, but you’re free to test other values
    Args:
        H: homography matrix as array (3, 3)
        pts1: interest points in img1, array (N, 2)
        pts2: interest points in img2, array (N, 2)
        threshold: scale down threshold
    
    Returns:
        number of inliers
    """

    distance = []

    pts1_trans = transform_pts(pts1, H)
    for i in range(pts2.shape[0]):
        dis = np.sqrt((pts1_trans[i][0] - pts2[i][0]) ** 2 + (pts1_trans[i][1] - pts2[i][1]) ** 2)  # L2-Distance
        if dis <= threshold:
            distance.append(dis)

    return len(distance)


def ransac_iters(w=0.5, d=min_num_pairs(), z=0.99):
    """ Computes the required number of iterations for RANSAC.

    Args:
        w: probability that any given correspondence is valid
        d: minimum number of pairs
        z: total probability of success after all iterations
    
    Returns:
        minimum number of required iterations
    """
    k = np.log10(1 - z) / np.log10(1 - w**d)

    return k


def ransac(pts1, pts2):
    """ RANSAC algorithm

    Args:
        pts1: matched points in img1, array (N, 2)
        pts2: matched points in img2, array (N, 2)
    
    Returns:
        best homography observed during RANSAC, array (3, 3)
    """

    num = 0
    threshold = 5
    k = ransac_iters(0.5, min_num_pairs(), 0.99)

    for j in range(int(k)):

        homo = compute_homography(pts1, pts2)  # calculation of the H matrix
        num_in = count_inliers(homo, pts1, pts2, threshold)

        if num_in > num:
            best_H = homo
            num = num_in

    return best_H


def find_matches(feats1, feats2, rT=0.8):
    """ Find pairs of corresponding interest points with distance comparsion
        Tips: We provide the default ratio value, but you’re free to test other values

    Args:
        feats1: SIFT descriptors of interest points in img1, array (N, 128)
        feats2: SIFT descriptors of interest points in img2, array (M, 128)
        rT: Ratio of similar distances
    
    Returns:
        idx1: list of indices of matching points in img1
        idx2: list of indices of matching points in img2
    """
    idx1 = []
    idx2 = []

    feats1 = np.array([f / np.linalg.norm(f) for f in feats1])
    feats2 = np.array([f / np.linalg.norm(f) for f in feats2])

    for i in range(feats1.shape[0]):
        dis = []
        for j in range(feats2.shape[0]):
            distance = np.sqrt(sum((feats1[i] - feats2[j]) ** 2))
            dis.append((j, distance))  # tuple. first: the indices of img_2, second: L2 distance
        dis = sorted(dis, key=lambda x: x[1])[:2]  # according to the second item in tuple sorting the list
        d = dis[0][1] / dis[1][1]  # d* criterion
        if d <= rT:
            idx1.append(i)
            idx2.append(dis[0][0])

    return idx1, idx2


def final_homography(pts1, pts2, feats1, feats2):
    """ re-estimate the homography based on all inliers

    Args:
       pts1: the coordinates of interest points in img1, array (N, 2)
       pts2: the coordinates of interest points in img2, array (M, 2)
       feats1: SIFT descriptors of interest points in img1, array (N, 128)
       feats2: SIFT descriptors of interest points in img1, array (M, 128)
    
    Returns:
        ransac_return: refitted homography matrix from ransac fucation, array (3, 3)
        idxs1: list of matched points in image 1
        idxs2: list of matched points in image 2
    """

    """
    # Normalization
    # expanding the original matrices in s-dimensional matrices
    
    pts1_new = np.ones((pts1.shape[0], 3))
    pts1_new[:, :2] = pts1
    pts2_new = np.ones((pts2.shape[0], 3))
    pts2_new[:, :2] = pts2

    # calculation the means of points 1 and 2
    mean_1 = np.mean(pts1, axis=0)
    mean_2 = np.mean(pts2, axis=0)

    # calculation the maximal norm of points 1
    s = 0
    for i in range(pts1.shape[0]):
        s_ = np.linalg.norm(pts1[i])
        if s_ > s:
            s = s_
    s_1 = s_

    # calculation the maximal norm of points 2
    s = 0
    for i in range(pts2.shape[0]):
        s_ = np.linalg.norm(pts2[i])
        if s_ > s:
            s = s_
    s_2 = s_

    # calculation the matrix t of points 1 and 2
    t_1 = np.array([[1 / s_1, 0, -mean_1[0] / s_1], [0, 1 / s_1, -mean_1[1] / s_1], [0, 0, 1]])
    t_2 = np.array([[1 / s_2, 0, -mean_2[0] / s_2], [0, 1 / s_2, -mean_2[1] / s_2], [0, 0, 1]])

    # normalization in interval [-1, 1]
    pts1_norm = t_1.dot(pts1_new.T).T
    pts2_norm = t_2.dot(pts2_new.T).T

    pts1_norm = pts1_norm[:, :2]
    pts2_norm = pts2_norm[:, :2]"""

    # in this task, it's not necessary to normalize the img points.
    idx_1, idx_2 = find_matches(feats1, feats2, 0.8)
    pts1_sub, pts2_sub = pts1[idx_1, :], pts2[idx_2, :]
    homo_matrix = ransac(pts1_sub, pts2_sub)

    return homo_matrix, idx_1, idx_2

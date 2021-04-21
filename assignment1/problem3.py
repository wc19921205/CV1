import numpy as np

def load_points(path):
    '''
    Load points from path pointing to a numpy binary file (.npy). 
    Image points are saved in 'image'
    Object points are saved in 'world'

    Returns:
        image: A Nx2 array of 2D points form image coordinate 
        world: A N*3 array of 3D points form world coordinate
    '''

    image_pts = np.empty((100, 3))
    world_pts = np.empty((100, 4))

    res = np.load(path)
    image_pts = res['image']
    world_pts = res['world']
    #print(image_pts)
    #print(world_pts)

    # sanity checks
    assert image_pts.shape[0] == world_pts.shape[0]

    # homogeneous coordinates
    assert image_pts.shape[1] == 3 and world_pts.shape[1] == 4
    return image_pts, world_pts


def create_A(x, X):
    """Creates (2*N, 12) matrix A from 2D/3D correspondences
    that comes from cross-product
    
    Args:
        x and X: N 2D and 3D point correspondences (homogeneous)
        
    Returns:
        A: (2*N, 12) matrix A
    """

    N, _ = x.shape
    assert N == X.shape[0]

    A = np.empty((2*N, 12))

    for i in range(0, 2*N, 2):  # step equal 2 and according to the formula in PPT
        #print(i//2)
        A[i][0:4] = 0
        A[i][4:8] = -np.array(X[i//2])
        A[i][8:12] = x[i//2][1] * np.array(X[i//2])

        A[i+1][0:4] = np.array(X[i//2])
        A[i+1][4:8] = 0
        A[i+1][8:12] = -x[i//2][0] * np.array(X[i//2])
        
    #print(A.shape)
    assert A.shape[0] == 2*N and A.shape[1] == 12
    return A


def homogeneous_Ax(A):
    """Solve homogeneous least squares problem (Ax = 0, s.t. norm(x) == 0),
    using SVD decomposition as in the lecture.

    Args:
        A: (2*N, 12) matrix A
    
    Returns:
        P: (3, 4) projection matrix P
    """

    u, s, v = np.linalg.svd(A, full_matrices=False)

    idx = np.argsort(s)[0]  # sorted array of eigenvalues, only the smallest one(index) remains.
    #print(u.shape, s.shape, v.shape)

    right_eigen_vector = v[:, idx]  # matrix remains, which corresponds to the smallest eigenvalue
    #right_eigen_vector = np.array([0.00191854517446,-0.00174998332538,0.00009670477631,-0.63861687292509, 0.00015184786215,0.00005945815592,
    #                                0.00252273614700,-0.76951593410321, -0.00000042931672,-0.00000093480944,0.00000027152848,-0.00075721579556])
    
    p = right_eigen_vector.reshape((3,4))
    return p


def solve_KR(P):
    """Using th RQ-decomposition find K and R 
    from the projection matrix P.
    Hint 1: you might find scipy.linalg useful here.
    Hint 2: recall that K has 1 in the the bottom right corner.
    Hint 3: RQ decomposition is not unique (up to a column sign).
    Ensure positive element in K by inverting the sign in K columns 
    and doing so correspondingly in R.

    Args:
        P: 3x4 projection matrix.
    
    Returns:
        K: 3x3 matrix with intrinsics
        R: 3x3 rotation matrix 
    """
    tmp = P[:, 0:3]
    #print()
    q, r = np.linalg.qr(tmp, mode='complete')

    test = np.matmul(q, r)

    return -r, -q


def solve_c(P):
    """Find the camera center coordinate from P
    by finding the nullspace of P with SVD.

    Args:
        P: 3x4 projection matrix
    
    Returns:
        c: 3x1 camera center coordinate in the world frame
    """
    u, s, v = np.linalg.svd(P, full_matrices=False)

    idx = np.argsort(s)[0]

    right_eigen_vector = v[:, idx]

    return right_eigen_vector
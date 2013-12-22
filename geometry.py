import numpy as np

def pr(x):
    '''Project a homogeneous vector or matrix. In the latter case each
    *row* will be interpreted as a vector to be projected.'''
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:-1] / x[-1]
    elif x.ndim == 2:
        return x[:,:-1] / x[:,[-1]]
    else:
        raise Exception, 'Cannot pr() an array with %d dimensions' % x.ndim

def unpr(x):
    '''Unproject a vector or matrix. In the latter case each *row* will be
    interpreted as a separate vector.'''
    x = np.asarray(x)
    if x.ndim == 1:
        return np.hstack((x, 1.))
    elif x.ndim == 2:
        return np.hstack((x, np.ones((len(x), 1))))
    else:
        raise Exception, 'Cannot unpr() an array with %d dimensions' % x.ndim

def prdot(H, X):
    '''Project a point through a homogeneous transformation,
    projecting and unprojecting automatically if necessary.'''
    H = np.asarray(H)
    X = np.asarray(X)
    assert np.ndim(H) == 2, 'The shape of H was %s' % str(H.shape)
    if X.ndim == 1:
        assert len(X) == np.size(H,1)-1, \
            'H.shape was %s, X.shape was %s' % (str(H.shape), str(X.shape))
        return pr(np.dot(H, unpr(X)))
    elif X.ndim == 2:
        assert np.size(X,1) == np.size(H,1)-1, \
            'H.shape was %s, X.shape was %s' % (str(H.shape), str(X.shape))
        return pr(np.dot(unpr(X), H.T))

def find_point_within_distance(x, points, distance):
    if len(points) > 0:
        x = np.asarray(x)
        points = np.asarray(points)
        distances = np.sqrt(np.sum(np.square(points - x), axis=1))
        i = np.argmin(distances)
        if distances[i] < distance:
            return i, points[i]
    return None, None

def in_bounds(p, shape):
    p = np.asarray(p)
    return np.all(p >= 0) and np.all(p < shape)

def fit_homography(fp, tp):
    fp = unpr(np.asarray(fp, float)).T
    tp = unpr(np.asarray(tp, float)).T
    return fit_homography_homogeneous(fp, tp)

def fit_homography_homogeneous(fp, tp):
    """ find homography H, such that fp is mapped to tp
        using the linear DLT method. Points are conditioned
        automatically."""

    fp = np.asarray(fp, float)
    tp = np.asarray(tp, float)

    if fp.shape[0] != 3:
        raise RuntimeError, 'number of rows in fp must be 3 (there were %d)' % fp.shape[0]

    if tp.shape[0] != 3:
        raise RuntimeError, 'number of rows in tp must be 3 (there were %d)' % tp.shape[0]

    if fp.shape[1] != tp.shape[1]:
        raise RuntimeError, 'number of points do not match'

    #condition points (important for numerical reasons)
    #--from points--
    m = np.mean(fp[:2], axis=1)
    maxstd = max(np.std(fp[:2], axis=1))
    if abs(maxstd) < 1e-8:
        # This is a degenerate configuration
        raise np.linalg.Np.LinalgError

    C1 = np.diag([1/maxstd, 1/maxstd, 1]) 
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp = np.dot(C1,fp)

    #--to points--
    m = np.mean(tp[:2], axis=1)
    #C2 = C1.copy() #must use same scaling for both point sets
    maxstd = max(np.std(tp[:2], axis=1))
    if abs(maxstd) < 1e-8:
        # This is a degenerate configuration
        raise np.linalg.Np.LinalgError

    C2 = np.diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp = np.dot(C2,tp)

    #create matrix for linear method, 2 rows for each correspondence pair
    nbr_correspondences = fp.shape[1]
    A = np.zeros((2*nbr_correspondences,9))
    for i in range(nbr_correspondences):        
        A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0,tp[0][i]*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]]
        A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1,tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],tp[1][i]]

    U,S,V = np.linalg.svd(A)

    H = V[8].reshape((3,3))    

    #decondition and return
    return np.dot(np.linalg.inv(C2), np.dot(H,C1))

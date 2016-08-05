import numpy as np

#batch gradient descent
def Gradient_Descent_Batch(X,y,theta,alpha):
    """batch gradient descent
    :type X:    numpy.array
    :param X:   a dataset(m*n)
    :type y:    numpy.array
    :param y:   a labels(m*1)
    """
    #  insert one column with 1
    X=np.insert(X,0,1,axis=1)
    m=X.shape[0]
    n=X.shape[1]

    max_iter=1000

    for num in range(0,max_iter):
        grad=X.T.dot((X.dot(theta)-y))

        old_theta=theta
        theta=theta-alpha*grad
        diff_theta=theta-old_theta
        if(np.sqrt(diff_theta.T.dot(diff_theta))<0.000000001):
            break

    return theta

# normal equation
def Normal_Equation(X,y):
    """normal equation
    :type X:    numpy.array
    :param X:   a dataset(m*n)
    :type y:    numpy.array
    :param y:   a labels(m*1)
    """
    X=np.insert(X,0,1,axis=1)
    X_mat=np.matrix(X)
    tmp=(X_mat.T*X_mat).I*X_mat.T
    tmp=np.array(tmp)
    theta=tmp.dot(y)
    theta=np.asarray(theta)
    return theta

# newton's method
def Newton_Method(X,y):
    """newton method
    :type X:    numpy.array
    :param X:   a dataset(m*n)
    :type y:    numpy.array
    :param y:   a labels(m*1)
    """
    X=np.insert(X,0,1,axis=1)
    m=X.shape[0]
    n=X.shape[1]

    # init theta
    theta=np.zeros((n,1))
    max_iter=100

    # ndarray 转 matrix
    X=np.matrix(X)
    y=np.matrix(y)
    theta=np.matrix(theta)

    for num in range(0,max_iter):
        old_theta=theta
        grad=X.T*(X*theta-y)
        H=X.T*X
        theta=theta-H.I*grad
        diff_theta=theta-old_theta
        if(np.sqrt(diff_theta.T*diff_theta)<0.000001):
            break

    return theta

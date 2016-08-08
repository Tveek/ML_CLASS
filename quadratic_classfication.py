# encoding=utf8
import numpy as np
import matplotlib.pyplot as plt

# calculate the sigmoid function
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def Gradient_Descent_Batch(X,y,theta,alpha):
    """batch gradient descent
    :type X:    numpy.array
    :param X:   a dataset(m*n)
    :type y:    numpy.array
    :param y:   a labels(m*1)
    """
    #  insert one column with 1
    # X=np.insert(X,0,1,axis=1)
    m=X.shape[0]
    n=X.shape[1]

    max_iter=1000

    for num in range(0,max_iter):
        error=sigmoid(X.dot(theta))-y
        old_theta=theta
        theta=theta-alpha*X.T.dot(error)
        diff_theta=theta-old_theta
        if(np.sqrt(diff_theta.T.dot(diff_theta))<0.0001):
            break

    return theta


def load_data():
    """加载原始数据，以及特征数据构建和组合
    　　h=theta0*1+theta1*x+theta2*x^2+theta3*y+theta4*y^2
    Returns:
        points:ndarray类型,(num_samples*2);被分类的点的坐标
        y:ndarray类型,(num_samples*1);被分类的点的标签
        X:ndarray类型，(num_samples*5);重新构建的特征数据
    """
    data=np.loadtxt('./data/quadratic_nonlinear_data.txt')
    points=data[:,:-1]
    lables=data[:,-1:]
    # 在点中插入一列1
    X=np.insert(points,0,1,axis=1)
    # 在第三，第五列插入相应的特征数据
    X=np.insert(X,2,points[:,0]**2,axis=1)
    X=np.insert(X,4,points[:,1]**2,axis=1)

    return points,X, lables

def plot(points,y,theta):
    """画出二次曲线分类的图
    Args:
        points:ndarray类型,(num_samples*2);被分类的点的坐标
        y:ndarray类型,(num_samples*1);被分类的点的标签
        theta:ndarray类型,(5*1);学习到的参数
    """
    # 获得sanmple的个数
    N=points.shape[0]
    # 按类显示
    for i in xrange(N):
        if int(y[i, 0]) == 0:
            plt.plot(points[i, 0], points[i, 1], 'ro')
        elif int(y[i, 0]) == 1:
            plt.plot(points[i, 0], points[i, 1], 'ko')

    # 按隐式函数的结果来显示学习到的二次曲线
    x = np.linspace(-2., 4.)
    y = np.linspace(-2., 4.)[:, None]
    plt.contour(x, y.ravel(), theta[0,0]+theta[1,0]*x+theta[2,0]*x**2+theta[3,0]*y+theta[4,0]*y**2, [0])

    # 坐标轴等比例显示
    plt.axis('equal')
    # plt.axis([-3,5,-2,4])
    plt.show()


if __name__ == '__main__':
    points,X,y=load_data()

    init_theta=np.zeros((5,1))
    alpha=0.001

    theta=Gradient_Descent_Batch(X,y,init_theta,alpha)
    # print theta
    plot(points,y,theta)

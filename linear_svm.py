# encoding=utf8
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def load_data():
    """加载原始数据，以及特征数据构建和组合
    　　h=theta0*1+theta1*x+theta2*x^2+theta3*y+theta4*y^2
    Returns:
        xy:ndarray类型,(num_samples*2);被分类的点的坐标
        lable:ndarray类型,(num_samples,);被分类的点的标签
    """
    class_data=np.loadtxt('./data/twofeature.txt')
    # 点的xy坐标
    xy=class_data[:,1:]
    # 标签数据
    lable=class_data[:,0:1]
    lable=lable.reshape(lable.shape[0],)

    return xy,lable


def plot(xy,lable,w,b,sv):
    """画出分类的数据，支持向量以及分类直线
       注意:无论形状，同一颜色为同一类；其中五角星形状的为该分类的支持向量
    Args:
        xy:ndarray类型,(num_samples*2);被分类的点的坐标
        lable:ndarray类型,(num_samples,);被分类的点的标签(1或-1)
        w:ndarray类型,(1*2);学习到的参数
        b:ndarray类型，(1*1);截距
        sv:ndarray类型,(num_sv*2);支持向量
    """
    num=xy.shape[0]
    for i in xrange(num):
        if int(lable[i]) == -1:
            # 判断是否为支持向量
            if xy[i].tolist() in sv.tolist():
                # 用符号和大小显著表明支持向量
                plt.plot(xy[i, 0], xy[i, 1], 'r*',markersize=10)
            else:
                plt.plot(xy[i, 0], xy[i, 1], 'ro')
        elif int(lable[i]) == 1:
            if xy[i].tolist() in sv.tolist():
                plt.plot(xy[i, 0], xy[i, 1], 'b*',markersize=10)
            else:
                plt.plot(xy[i, 0], xy[i, 1], 'bo')


    # 画分类直线
    x = np.linspace(0, 4.)
    y = np.linspace(1.5, 5.)[:, None]
    plt.contour(x, y.ravel(), w[0,0]*x+w[0,1]*y+b, [0],colors='y')

    # 坐标轴等比例显示
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    # 加载数据
    X,y=load_data()
    # 使用scikit-learn SVC(surppot vector classfication)库进行SVM分类
    clf=SVC(C=1,kernel='linear')
    clf.fit(X,y)

    # 获得参数ｗ,该方法限制linear模型
    w=clf.coef_
    # 这个适用更加广阔
    # w=np.dot(clf.dual_coef_,clf.support_vectors_)

    # 截距
    b=clf.intercept_

    sv=clf.support_vectors_
    # print np.dot(sv,w.T)+b
    # print sv

    # 画图
    plot(X,y,w,b,sv)

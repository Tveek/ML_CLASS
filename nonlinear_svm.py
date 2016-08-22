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
    class_data=np.loadtxt('./data/nonlinear_svmdata.txt')
    # 点的xy坐标
    xy=class_data[:,1:]
    # 标签数据
    lable=class_data[:,0:1]
    lable=lable.reshape(lable.shape[0],)

    return xy,lable


def plotboundary(labels, features, model, varargin=''):
    """画出分类的数据和分类曲线
    Args:
        lables:ndarray类型,(num_samples,);被分类的点的标签(1或-1)
        features:ndarray类型,(num_samples*2);被分类的点的坐标
        model:SVC类型;训练的模型
        varargin:str类型;可选变量,控制是否显示分类正确的程度(由颜色区分),默认为空(不显示),等于't'(显示)
    """
    xplot = np.linspace(min(features[:,0]), max(features[:,0]), 100)
    yplot = np.linspace(min(features[:,1]), max(features[:,1]), 100)
    # 构建网格数据
    [X,Y]=np.meshgrid(xplot,yplot)

    vals = np.zeros(X.shape)



    for i in range(0,X.shape[1]):
        # 重构画图数据
        x=np.array([X[:,i],Y[:,i]]).T
        # 获得决策值
        decision_values=model.decision_function(x)
        vals[:,i]=decision_values

    # 判断是否显示分类正确的程度
    if(varargin=='t'):
       plt.contourf(X,Y,vals,50,linestyles=None)

    plt.contour(X,Y,vals,[0],linestyles='solid',linewidths=2,colors='black')

    # 找到不同样本的行数
    pos=np.where(labels==1)
    neg=np.where(labels==-1)

    # 按类别画分类点
    plt.plot(features[pos,0],features[pos,1],'bo')
    plt.plot(features[neg,0],features[neg,1],'ro')


    plt.show()


if __name__ == '__main__':

    # 加载数据
    X,y=load_data()
    # 创建svc对象
    clf=SVC(C=1,kernel='rbf',gamma=100)

    clf.fit(X,y)

    plotboundary(y,X,clf,'t')

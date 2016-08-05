# encoding=utf8
import numpy as np
import matplotlib.pyplot as plt
import random


class Perceptron:
    def __init__(self):
        self.datas,self.lables=self.load_data()
        self.num_samples,self.num_features=self.datas.shape
        self.w=np.ones((self.num_features,1))

    def calculate_error(self,vector):
        """计算错误点的个数
        Args:
            vector:ndarray类型,;表示学到的theta
        Returns:
            false_data:int类型;在学到某个theta的情况下，被分类错误的点的个数
            theta:ndarray类型,(num_feature*10);表示更新后的theta
        """
        X=self.datas
        y=self.lables
        w=vector
        N=self.num_samples
        false_data=0
        for i in range(0,N):
            if(np.sign(np.dot(X[i,:],w))!=y[i][0]):
                false_data+=1
        return false_data


    # 数据线性可分
    def navie_pla(self):
        """数据线性可分下的pla算法
        """
        X=self.datas
        y=self.lables
        w=self.w
        N=self.num_samples

        while True:
            false_data=0
            for i in range(0,N):

                if(np.sign(np.dot(X[i,:],w))!=y[i][0]):
                    x=X[i,:].reshape(3,1)
                    w+=y[i][0]*x
                    false_data+=1
            if(false_data==0):
                break
        self.w=w

    # 数据有杂音,不完全线性可分
    def pocket_pla(self,max_iter):
        """计算错误点的个数
        Args:
            max_iter:int类型,;最大迭代次数
        Returns:
            false_data:int类型;在学到某个theta的情况下，被分类错误的点的个数
            theta:ndarray类型,(num_feature*10);表示更新后的theta
        """
        X=self.datas
        y=self.lables
        w=self.w
        N=self.num_samples
        false_data=self.calculate_error(w)

        for i in range(0,max_iter):
            rand_sort = range(N)
            rand_sort = random.sample(rand_sort,N)
            for j in rand_sort:
                if(np.sign(np.dot(X[j,:],w))!=y[j][0]):
                    x=X[j,:].reshape(3,1)
                    tmp_w=w+0.01*y[j][0]*x
                    tmp_false_data=self.calculate_error(tmp_w)
                    # print tmp_false_data
                    if(tmp_false_data<=false_data):
                        w=tmp_w
                        false_data=tmp_false_data
                        # print false_data
                    break
        # print false_data
        self.w=w



    def load_data(self):
        data=np.loadtxt('./data/pla_nonlinear_data.txt')
        points=data[:,:-1]
        lables=data[:,-1:]
        # 在点中插入一列1
        points=np.insert(points,0,1,axis=1)
        return points, lables

    def plot2D(self):
        xy=self.datas[:,1:]
        lable=self.lables
        N=self.num_samples

        # 画点
        for i in xrange(N):
            if int(lable[i, 0]) == -1:
                plt.plot(xy[i, 0], xy[i, 1], 'ro')
            elif int(lable[i, 0]) == 1:
                plt.plot(xy[i, 0], xy[i, 1], 'bo')
        # 画线
        theta=self.w.reshape((3,))

        min_x = min(xy[:, 0])
        max_x = max(xy[:, 0])
        y_min_x = float(-theta[0] - theta[1] * min_x) / theta[2]
        y_max_x = float(-theta[0] - theta[1] * max_x) / theta[2]
        plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')


        # plt.axis([-6,6,-6,6])
        plt.xlabel('x'); plt.ylabel('y')
        plt.show()

if __name__ == '__main__':


    p=Perceptron()
    p.pocket_pla(1000)
    # p.navie_pla()
    p.plot2D()

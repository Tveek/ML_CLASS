#encoding=utf8
import numpy as np
from PIL import Image

class FaceRecognition(object):
    """利用PCA算法，实现简单的人脸识别"""
    def __init__(self, image_size,image_num,image_path):
        """类的初始化
        Args:
            image_size:  整数;    表示一张人脸总像素点个数
            image_num:   整数;    表示训练用到的人脸的个数
            image_path:  字符串;   训练的人脸存放的文件夹位置
        """
        self.image_size=image_size
        self.image_num=image_num
        self.image_path=image_path
    def load_data(self):
        """加载训练数据
        Returns:
            image_data: ndarray, (image_size*image_num);    所有人脸的灰度图数据，每一列表示一张人脸
        """
        image_data=np.zeros((self.image_size,self.image_num))
        for i in range(0,self.image_num):
            im=Image.open(self.image_path+str(i+1)+".jpg")
            im=im.convert("L")
            data = im.getdata()
            data = np.array(data)
            image_data[:,i]=data

        return image_data
    def compute_mean_diff(self,image_array):
        """计算每个像素点的均值，并将每一列(一张人脸)的值减去相应像素点的均值
        Args:
            image_array: ndarray, (image_size*image_num);   所有人脸的灰度图数据，每一列表示一张人脸
        Returns:
            mean_diff:   ndarray, (image_size*image_num);   减去各个像素点均值的人脸矩阵
            mean:        ndarray, (image_size,);            各个像素点均值
        """
        mean=np.mean(image_array,axis=1)
        temp=image_array.T-mean
        mean_diff=temp.T

        return mean_diff,mean
    def eigenfaces(self,mean_diff):
        """计算每个像素点的均值，并将每一列(一张人脸)的值减去相应像素点的均值
        Args:
            mean_diff:   ndarray, (image_size*image_num);   减去各个像素点均值的人脸矩阵
        Returns:
            eigenfaces:   ndarray, (image_num*k);           特征向量构成的特征脸矩阵，k是最终选择的特征向量个数
        """
        covariance=np.dot(mean_diff.T,mean_diff)
        eigenvalues,eigenvectors=np.linalg.eig(covariance)
        # 下面注释的代码课实现特征值和特征矩阵按大到小排序
        # idx=np.argsort(-eigenvalues)
        # eigenvalues=eigenvalues[idx]
        # eigenvectors=eigenvectors[:,idx]
        idx=np.where(eigenvalues>1)
        eigenvectors=eigenvectors[:,idx[0]]
        eigenfaces=np.dot(mean_diff,eigenvectors)

        return eigenfaces
    def recognition(self,train,test,eigenfaces):
        """计算每个像素点的均值，并将每一列(一张人脸)的值减去相应像素点的均值
        Args:
            train:        ndarray, (image_size*image_num)   训练人脸矩阵
            test:         ndarray, (image_size,)            测试人脸矩阵
            eigenfaces:   ndarray, (image_num*k);           特征向量构成的特征脸矩阵，k是最终选择的特征向量个数
        Returns:
            index:        整数,                              检测人脸是训练人脸库中的第几张人脸
        """
        # 将训练和测试数据映射到eigenfaces空间
        projected=np.dot(train.T,eigenfaces)
        projected_test=np.dot(test.T,eigenfaces)
        # 计算测试人脸和训练库中的各张人脸的欧拉距离
        euler_distance=np.sum((projected-projected_test)**2,axis=1)**0.5
        # 选择欧拉距离最小的作为人脸识别的结果
        recognized_index=np.where(euler_distance==np.min(euler_distance))
        index=recognized_index[0][0]+1

        return index


if __name__ == '__main__':
    # 训练人脸路径
    train_image_path="./data/pca_train_data/"
    # 构建识别类，并计算出特征脸和像素点均值
    recognition=FaceRecognition(180*200,20,train_image_path)
    train_grey_image=recognition.load_data()
    train_diff_image,mean=recognition.compute_mean_diff(train_grey_image)
    eigenfaces=recognition.eigenfaces(train_diff_image)
    # 测试数据构建
    test_image="./data/pca_test_data/10.jpg"　　#修改数值10测试其他人脸是否识别准确.
    im=Image.open(test_image)
    im=im.convert("L")
    temp_data = im.getdata()
    temp_data = np.array(temp_data)
    test_diff_data=temp_data-mean

    # 识别人脸
    index=recognition.recognition(train_diff_image,test_diff_data,eigenfaces)
    # 显示结果
    im=Image.open("./data/pca_train_data/"+str(index)+".jpg")
    im.show()

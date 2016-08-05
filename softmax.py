# encoding=utf8
import numpy as np
import scipy.sparse
import scipy.optimize
import struct


class SoftMax(object):
    """docstring for """
    def __init__(self, num_feature, num_class, lamda):
        """构成函数
        Args:
            num_feature:int类型;表示每张图片特征点的个数
            num_class:int类型;表示总共有多少类别
            lamda:float类型;衰减因子
        """
        self.num_feature=num_feature
        self.num_class=num_class
        self.lamda=lamda
        # 初始化theta
        self.theta=np.zeros((num_feature,num_class))
        # self.theta=np.ones((num_feature*num_class,1))

    def ground_truth(self,lables):
        """将标签转换成ground truth矩阵
        Args:
            lables:ndarray类型(num_sample*1);表示mnist每张图像对应的标签
        Returns:
            dataset:ndarray类型,(num_sample*1０);矩阵的行代表每一个样本，列表示从０～９，矩阵的值位１表示该样本的值是１对应的列数
        """
        # 准备构造数据
        num_sample=lables.shape[0]
        col=lables.flatten()
        row=np.arange(num_sample)
        data=np.ones(num_sample)
        # 利用库构造矩阵
        ground_truth=scipy.sparse.csr_matrix((data,(row,col)))
        # 转换成稀疏矩阵,同时转成ndarray
        ground_truth=np.array(ground_truth.todense())
        return ground_truth

    def softmax_h(self,theta,input):
        """获得假设函数h
        Args:
            theta:ndarray类型,(num_feature*10);表示theta
            input:ndarray类型，(num_sample*num_feature);表示输入数据
        Returns:
            ｈ:ndarray类型,(num_sample*1０);矩阵的每行对应的是该样本是０～９(就是第几列)的概率分别是多少
        """
        num_sample=input.shape[0]
        m=np.dot(input,theta)
        # 防止数据overflow,对ｍ进行预处理，已证明对假设函数无影响
        m-=np.max(m,1).reshape(num_sample,1)
        e=np.exp(m)
        # 假设函数h
        h=e/np.sum(e,1).reshape(num_sample,1)
        return h

    def softmax_cost(self,theta,input,lables):
        """构建损失函数
        Args:
            theta:ndarray类型,(num_feature*10);表示theta
            input:ndarray类型，(num_sample*num_feature);表示输入数据
            lables:ndarray类型(num_sample*1);表示mnist每张图像对应的标签
        Returns:
            cost:float类型;表示总的损失
            theta:ndarray类型,(num_feature*10);表示更新后的theta
        """
        num_sample=lables.shape[0]
        # 获得样本对应每一个分类的标注信息矩阵(样本是该分类就为１，不是该类则为０)
        ground_truth=self.ground_truth(lables)
        #这个地方值得注意，优化器是将上一次得到的theta传进来，第一次是传初始化的theta
        theta=theta.reshape(self.num_feature,self.num_class)
        h=self.softmax_h(theta,input)


        # 损失函数
        cost=-(np.sum(ground_truth*np.log(h)))/num_sample+0.5*self.lamda*np.sum(theta**2)
        # 梯度

        grad_theta=-np.dot(input.T,ground_truth-h)/num_sample+self.lamda*theta

        # 为了满足优化器的要求，将其展开
        grad_theta=grad_theta.flatten()
        # print cost
        return [cost,grad_theta]

    def softmax_test(self,theta,input,lables):
        """构建损失函数
        Args:
            theta:ndarray类型,(num_feature*10);表示theta
            input:ndarray类型，(num_sample*num_feature);表示输入数据
            lables:ndarray类型(num_sample*1);表示mnist每张图像对应的标签
        Returns:
            accuracy:float类型;表示在测试数据上，学到的模型的正确率
        """
        num_sample=lables.shape[0]
        h=self.softmax_h(theta,input)
        # 这个函数有点意思，表示获得ndarray数组每行第一个最大元素所在列的位置
        # 在这里也就是我们每个测试样本对应的预测值
        res=np.argmax(h,axis=1).reshape(num_sample,1)

        accuracy=np.sum(res==lables)/float(num_sample)
        return accuracy

########################################
def load_mnist_image(filename):
    """获取mnist图像的特征信息
    Args:
        filename:str类型;表示mnist图像数据的文件名
    Returns:
        dataset:ndarray类型,(num_images*784);矩阵的每一行代表一个图像信息
    """
    # 读二进制文件
    image_file = open(filename, 'rb')
    buf=image_file.read()
    image_file.close()
    # 读取前4个 32 bit integer,因为它们是数据的描述信息
    index = 0
    magic, num_images , num_rows , num_cols = struct.unpack_from('>IIII' , buf , index)
    # 获取图像的特征点个数
    num_feature=num_rows*num_cols
    index += struct.calcsize('>IIII')


    # 初始化存储数据的矩阵
    dataset=np.zeros((num_images,num_feature))
    for i in range(0,num_images):
        # 读入一张图片的数据,28*28
        im = struct.unpack_from('>784B' ,buf, index)
        index += struct.calcsize('>784B')
        im=np.array(im)
        # 填充数据
        dataset[i,:]=im

    # 数据归一化
    dataset=dataset/255
    return dataset

def load_mnist_labels(filename):
    """获取mnist图像标签的数据
    Args:
        filename:str类型;表示mnist图像标签的文件名
    Returns:
        dataset:ndarray类型,(num_images*1);矩阵的每一行代表一个图像表示的数值
    """
    # 读二进制文件
    image_file = open(filename, 'rb')
    buf=image_file.read()
    image_file.close()
    # 读取前4个 32 bit integer,因为它们是数据的描述信息
    index = 0
    magic, num_labels = struct.unpack_from('>II' , buf , index)
    # 获取图像的特征点个数

    index += struct.calcsize('>II')
    # >60000B 代表按大端取60000个byte数据
    lens='>'+str(num_labels)+'B'
    labels=struct.unpack_from(lens ,buf, index)
    labels=np.array(labels).reshape(num_labels,1)

    return labels


# 初始化各种参数
num_feature=784
num_class=10
lamda=1e-4
max_iterations = 100

#########################################
# 训练


# 加载训练数据
train_images   = load_mnist_image('./data/train-images.idx3-ubyte')
train_labels = load_mnist_labels('./data/train-labels.idx1-ubyte')

# 初始化softmax模型
softmax_model=SoftMax(num_feature,num_class,lamda)

# 利用优化器迭代数据
opt_solution  = scipy.optimize.minimize(softmax_model.softmax_cost, softmax_model.theta,
                                            args = (train_images, train_labels), method = 'L-BFGS-B',
                                            jac = True, options = {'maxiter': max_iterations})

# 得到优化后的theta
opt_theta = opt_solution.x.reshape(num_feature,num_class)

###########################################
# 测试


#加载测试数据
test_images   = load_mnist_image('./data/t10k-images.idx3-ubyte')
test_labels = load_mnist_labels('./data/t10k-labels.idx1-ubyte')

accuracy=softmax_model.softmax_test(opt_theta,test_images,test_labels)

print accuracy

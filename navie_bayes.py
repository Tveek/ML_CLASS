# encoding=utf8
from __future__ import division
import numpy as np
import scipy.sparse


def load_data():
    """加载训练和测试数据并处理
    Returns:
        train_features:ndarray类型,(num_sample*2500);每行词典中的单词分别在一封邮件中出现的次数
        train_lables:ndarray类型,(num_sample,);每个元素表示对应的邮件是否为垃圾邮件,用0和1表示.
        test_features:同上
        test_lables:同上

    """
    # 加载数据
    train_data=np.loadtxt('./data/email_train_features.txt')
    train_lables=np.loadtxt('./data/email_train_labels.txt')
    test_data=np.loadtxt('./data/email_test_features.txt')
    test_lables=np.loadtxt('./data/email_test_labels.txt')

    # 获取行和列以及相应的值
    train_row=train_data[:,0]-1
    train_col=train_data[:,1]-1
    train_frequency=train_data[:,2]

    # 利用行,列的位置和值构建矩阵
    train_features=scipy.sparse.csr_matrix((train_frequency,(train_row,train_col)))

    test_row=test_data[:,0]-1
    test_col=test_data[:,1]-1
    test_frequency=test_data[:,2]
    test_features=scipy.sparse.csr_matrix((test_frequency,(test_row,test_col)))

    # 将构建的矩阵稀疏化
    train_features=np.array(train_features.todense())
    test_features=np.array(test_features.todense())

    return train_features,train_lables,test_features,test_lables

def navie_bayes(X,y):
    """
    Args:
        X:ndarray类型(num_sample*num_feature);样本特征矩阵
        y:ndarray类型(num_sample,);样本标签
    Returns:
        prob_tokens_spam:ndarray类型,(num_feature,);垃圾邮件中,每个特征所占的比例
        prob_tokens_nonspam:ndarray类型,(num_feature,);非垃圾邮件中,每个特征所占的比例
        prob_spam:flaot类型;所有邮件中,垃圾邮件所占比例
    """
    # 获得样本个数和特征个数
    num_sample=X.shape[0]
    num_feature=X.shape[1]

    # 垃圾邮件所在行数
    spam_index=np.array(np.where(y==1))[0]

    nonspam_index=np.array(np.where(y==0))[0]

    # 垃圾邮件占总邮件的概率
    prob_spam=len(spam_index)/num_sample

    # 在垃圾邮件中,计算训练样本各个特征所出现次数的总和
    wc_spam=np.sum(X[spam_index,:],axis=0)

    wc_nonspam=np.sum(X[nonspam_index,:],axis=0)

    # 求概率以及平滑处理
    prob_tokens_spam = (wc_spam + 1) / (sum(wc_spam) + num_feature)
    prob_tokens_nonspam = (wc_nonspam + 1) / (sum(wc_nonspam) + num_feature)

    return prob_tokens_spam,prob_tokens_nonspam,prob_spam


if __name__ == '__main__':
    train_features,train_lables,test_features,test_lables=load_data()
    prob_tokens_spam,prob_tokens_nonspam,prob_spam=navie_bayes(train_features,train_lables)

    # 求概率,进行了log处理
    test_spam_proc = np.sum(test_features*np.log(prob_tokens_spam),axis=1)+ np.log(prob_spam)
    test_nonspam_proc = np.sum(test_features*np.log(prob_tokens_nonspam),axis=1) + np.log(1-prob_spam)

    # 预测
    test_spam=test_spam_proc>test_nonspam_proc

    accuracy = np.sum(test_spam==test_lables) / len(test_lables)
    print accuracy

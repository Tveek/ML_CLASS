import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import time


# calculate the sigmoid function
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


# train a logistic regression model using some optional optimize algorithm
# input: train_x is a mat datatype, each row stands for one sample
#        train_y is mat datatype too, each row is the corresponding label
#        opts is optimize option include step and maximum number of iterations
def trainLogRegres(train_x, train_y, opts):
    # calculate training time
    startTime = time.time()

    numSamples, numFeatures = train_x.shape
    alpha = opts['alpha']; maxIter = opts['maxIter']
    theta = np.matlib.ones((numFeatures, 1))

    # optimize through gradient descent algorilthm
    for k in range(maxIter):
        if opts['optimizeType'] == 'gradDescent': # gradient descent algorilthm
            output = sigmoid(train_x * theta)
            error = train_y - output
            theta = theta + alpha * train_x.T * error
        elif opts['optimizeType'] == 'stocGradDescent': # stochastic gradient descent
            for i in range(numSamples):
                output = sigmoid(train_x[i, :] * theta)
                error = train_y[i, 0] - output
                theta = theta + alpha * train_x[i, :].T * error
        elif opts['optimizeType'] == 'smoothStocGradDescent': # smooth stochastic gradient descent
            # randomly select samples to optimize for reducing cycle fluctuations
            dataIndex = range(numSamples)
            for i in range(numSamples):
                alpha = 4.0 / (1.0 + k + i) + 0.01
                randIndex = int(np.random.uniform(0, len(dataIndex)))
                output = sigmoid(train_x[randIndex, :] * theta)
                error = train_y[randIndex, 0] - output
                theta = theta + alpha * train_x[randIndex, :].T * error
                del(dataIndex[randIndex]) # during one interation, delete the optimized sample
        else:
            raise NameError('Not support optimize method type!')


    print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
    return theta


# test your trained Logistic Regression model given test set
def testLogRegres(theta, test_x, test_y):
    numSamples, numFeatures = np.shape(test_x)
    matchCount = 0
    for i in xrange(numSamples):
        predict = sigmoid(test_x[i, :] * theta)[0, 0] > 0.5
        if predict == bool(test_y[i, 0]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy


# show your trained logistic regression model only available with 2-D data
def showLogRegres(theta, train_x, train_y):
    # notice: train_x and train_y is mat datatype
    numSamples, numFeatures = np.shape(train_x)
    if numFeatures != 3:
        print "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1

    # draw all samples
    for i in xrange(numSamples):
        if int(train_y[i, 0]) == 0:
            plt.plot(train_x[i, 1], train_x[i, 2], 'ro')
        elif int(train_y[i, 0]) == 1:
            plt.plot(train_x[i, 1], train_x[i, 2], 'bo')

    # draw the classify line
    min_x = min(train_x[:, 1])[0, 0]
    max_x = max(train_x[:, 1])[0, 0]
    theta = theta.getA()  # convert mat to array
    y_min_x = float(-theta[0] - theta[1] * min_x) / theta[2]
    y_max_x = float(-theta[0] - theta[1] * max_x) / theta[2]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

def loadData():
    data=np.loadtxt('./data/lr_nonlinear_data.txt')
    train_x=data[:,0:2]
    train_y=data[:,2:]
    train_x=np.insert(train_x,0,1,axis=1)

    return np.mat(train_x), np.mat(train_y)


## step 1: load data
print "step 1: load data..."
train_x, train_y = loadData()
test_x = train_x; test_y = train_y

## step 2: training...
print "step 2: training..."
opts = {'alpha': 0.01, 'maxIter': 500, 'optimizeType': 'smoothStocGradDescent'}
optimalTheta = trainLogRegres(train_x, train_y, opts)

## step 3: testing
print "step 3: testing..."
accuracy = testLogRegres(optimalTheta, test_x, test_y)

## step 4: show the result
print "step 4: show the result..."
print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
showLogRegres(optimalTheta, train_x, train_y)

print sigmoid(train_x*optimalTheta)

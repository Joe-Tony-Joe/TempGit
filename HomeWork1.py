from sklearn.datasets import load_iris
import numpy as np
import operator

np.random.seed(20210605)


def KNN(test, train, label, k):
    dataSize = train.shape[0]
    diffMat = np.tile(test, (dataSize, 1)) - train
    distances = (diffMat**2).sum(axis=1)**0.5
    sortedDist = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = label[sortedDist[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortClassCount[0][0]


def main():
    seq = np.random.permutation(np.arange(150))
    iris = load_iris()
    x = iris.data  # 特征向量，并且是按顺序排列的
    y = iris.target  # 标签
    x_train, x_test, y_train, y_test = x[seq[0:105], :], x[seq[105:-1], :], y[seq[0:105]], y[seq[105:-1]]
    predict = []
    for j in range(len(x_test)):
        pred = KNN(x_test[j], x_train, y_train, 3)
        predict.append(pred)
    predict = np.array(predict)
    y_test = np.array(y_test)
    accuracy = np.sum(predict == y_test)/len(y_test)
    print("accuracy is", accuracy)


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import time
import operator



Ptime = time.time()
mnist_data = fetch_openml("mnist_784", version=1, cache=True)
x, y = mnist_data["data"].to_numpy(), mnist_data["target"].to_numpy()
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=100)
Ntime = time.time()
print("数据加载时间为：%fs" %(Ntime - Ptime))


P2time = time.time()
mlp = MLPClassifier(solver='adam', alpha=1e-5, activation='relu',
                    hidden_layer_sizes=(60,36,12,10))
mlp.fit(train_x,train_y)
y_pred = mlp.predict(test_x)
N2time = time.time()

print("训练用时为:%fs" %(N2time-P2time))

print("准确率为: %f" %accuracy_score(test_y, y_pred)) # 这里y_pred为预测值


plt.figure(1)
for i in range(10):
    image = test_x[i]   # 得到包含第i张图的像素向量，为1*768
    pixels = image.reshape((28, 28)) # 将原始像素向量转换为28*28的像素矩阵
    plt.subplot(5,2,i+1)
    plt.imshow(pixels, cmap='gray')
    plt.title(y_pred[i])
    plt.axis('off')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45,
                    wspace=0.85)
plt.show()


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
    mnist_data = fetch_openml("mnist_784", version=1, cache=True)
    x, y = mnist_data["data"].to_numpy()[1:1000,:], mnist_data["target"].to_numpy()[1:1000]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=100)
    predict = []
    for j in range(len(test_y)):
        pred = KNN(test_x[j], train_x, train_y, 3)
        predict.append(pred)
    predict = np.array(predict)

    plt.figure(1)
    for i in range(10):
        image = test_x[i]  # 得到包含第i张图的像素向量，为1*768
        pixels = image.reshape((28, 28))  # 将原始像素向量转换为28*28的像素矩阵
        plt.subplot(5, 2, i + 1)
        plt.imshow(pixels, cmap='gray')
        plt.title(predict[i])
        plt.axis('off')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45,
                        wspace=0.85)
    plt.show()

    print("准确率为:%f" %accuracy_score(test_y, predict))

main()
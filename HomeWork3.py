import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
x = np.arange(-10,10,0.02)
y1 = np.sin(x) + 0.2 * np.random.rand(len(x))
y2 = x**2+3*x+1 + 5 * np.random.rand(len(x))


reg1 = MLPRegressor(solver='lbfgs', alpha=1e-5, activation='tanh', random_state=1000,
                    hidden_layer_sizes=(10,9,8),tol=0.001)
reg2 = MLPRegressor(solver='lbfgs', alpha=1e-5, activation='relu', random_state=1000,
                    hidden_layer_sizes=(9,8,4),shuffle=False,learning_rate='constant',tol=0.001,)
reg1.fit(x.reshape(-1,1),y1)
y1_pred = reg1.predict(x.reshape(-1,1))
reg2.fit(x.reshape(-1,1),y2)
y2_pred = reg2.predict(x.reshape(-1,1))

plt.figure(211)
train1, pred1 = plt.plot(x, y1, x, y1_pred)
plt.legend([train1,pred1],["train","test"])
plt.figure(212)
train2, pred2 = plt.plot(x, y2, x, y2_pred)
plt.legend([train2,pred2],["train","test"])
plt.show()

from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.3, random_state=100)
mlp = MLPClassifier(hidden_layer_sizes=(20,3), activation='relu', max_iter=10000)
mlp.fit(train_x,train_y)
y_pred = mlp.predict(test_x)
print("The accuracy is %f" % accuracy_score(test_y, y_pred))















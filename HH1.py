import numpy as np
from matplotlib import pyplot as plt

X1 = np.array([1,1,1]).transpose()
X2 = np.array([-3,4,1]).transpose()
X3 = np.array([2,-5,1]).transpose()

W1 = np.zeros(3)
W2 = np.zeros(3)
W3 = np.zeros(3)
p = 1
x1 = np.arange(-10,10,0.1)
y1 = np.arange(-10,10,0.1)
u1 = np.ones(len(x1))
XXX = np.vstack((x1,y1,u1))
XXX = np.matrix(XXX)
print(W1,W2,W3)
def G1(X):
    print("G1=",np.dot(W1.transpose(), X))
    return np.dot(W1.transpose(), X)

def G2(X):
    print("G2=", np.dot(W2.transpose(), X))
    return np.dot(W2.transpose(), X)

def G3(X):
    print("G3=", np.dot(W3.transpose(), X))
    return np.dot(W3.transpose(), X)

while(1):
    if G1(X1) == G2(X1) and G1(X1) == G3(X1):
        W1 = W1 + p * X1
        W2 = W2 - p * X1
        W3 = W3 - p * X1
    elif G1(X1) == G2(X1):
        W1 = W1 + p * X1
        W2 = W2 - p * X1
    elif G1(X1) == G3(X1):
        W1 = W1 + p * X1
        W3 = W3 - p * X1
    print(W1,W2,W3)
    if G2(X2) == G1(X2) and G2(X2) == G3(X2):
        W2 = W2 + p * X2
        W1 = W1 - p * X2
        W3 = W3 - p * X2
    elif G2(X2) == G1(X2):
        W2 = W2 + p * X2
        W1 = W1 - p * X2
    elif G2(X2) == G3(X2):
        W2 = W2 + p * X2
        W3 = W3 - p * X2
    print(W1, W2, W3)
    if G3(X3) == G1(X3) and G3(X3) == G2(X3):
        W3 = W3 + p * X3
        W1 = W1 - p * X3
        W2 = W2 - p * X3
    elif G3(X3) == G1(X3):
        W3 = W3 + p * X3
        W1 = W1 - p * X3
    elif G3(X3) == G2(X3):
        W3 = W3 + p * X3
        W2 = W2 - p * X3
    print(W1, W2, W3)
    if (G1(X1) != G2(X1)) and (G1(X1) != G3(X1)):
        if (G2(X2) != G1(X2)) and (G2(X2) != G3(X2)):
            if (G3(X3) != G1(X3)) and (G2(X3) != G3(X3)):
                break
d1 = W1 - W2
d2 = W1 - W3
d3 = W2 - W3
print(W1,W2,W3)
print("W1 = " , W1)
print("W2 = " , W2)
print("W3 = " , W3)
print("决策面D1 ", d1,".TX")
print("决策面D2 ", d2,".TX")
print("决策面D1 ", d3,".TX")
D1 = np.tile(d1,(len(x1),1))
D1 = np.matrix(D1)
D2 = np.tile(d2,(len(x1),1))
D2 = np.matrix(D2)
D3 = np.tile(d3,(len(x1),1))
D3 = np.matrix(D3)
z1 = np.dot(D1 , XXX)
z2 = np.dot(D2 , XXX)
z3 = np.dot(D3 , XXX)
x,y = np.meshgrid(x1, y1)
fig = plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, z1, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none',color='red')
fig = plt.figure(2)
bx = plt.axes(projection='3d')
bx.plot_surface(x, y, z2, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none',color='blue')
fig = plt.figure(3)
cx = plt.axes(projection='3d')
cx.plot_surface(x, y, z3, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none',color='yellow')
plt.show()
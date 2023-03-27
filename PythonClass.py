# # import mediapipe as mp
#
# # message = ' python '
# # print(message.rsplit())
# # print(message.lstrip())
# # print(message)
#
# # rlist = ['abs','孙大佬','89']
# # print(rlist)
# # print(rlist[0])
# # print(len(rlist))
# # print(rlist[-1])
#
# # message = ' my first is ' + rlist[0].title()
# # print(message)
#
# names = ['aaa', 'bbb', 'eee','ccc', 'ddd']
# # print(names)
# # for i in range(len(names)):
# #     message = "My name is "+ names[i].title()
# #     print(message)
#
# # names.append('sdl')
# # print(names)
# # names.insert(2, 33)
# # print(names)
# # names.pop()
# # print(names)
# # names.remove(33)
# # print(names)
# # print(names[0].upper().lstrip('A'))
# # names.append('aaa')
# # print(names)
# #
# # invite = ['sdl', 'dsa', 'erg', 'ggfd']
# # for i in range(len(invite)):
# #     msg = "Dear " + invite[i] + '\n \tWe invita you'
# #     print(msg)
# #
# # no_present = ['sdl', 'dsa']
# # re_invite = list(invite)
# # print(type(re_invite))
# # # print(re_invite)
# # # re_invite.remove(no_present[0])
# # # print(re_invite)
#
# print(names)
# # names.sort()
# # print(names)
# # names.sort(reverse=True)
# # print(names)
# # a = sorted(names,reverse=False)
# # print(a)
# names.reverse()
# print(names)
# names = names[-1:]
# print(names)
# dataSet =     [[0, 0, 0, 0, 'no'],  # 数据集
#                [0, 0, 0, 1, 'no'],
#                [0, 1, 0, 1, 'yes'],
#                [0, 1, 1, 0, 'yes'],
#                [0, 0, 0, 0, 'no'],
#                [1, 0, 0, 0, 'no'],
#                [1, 0, 0, 1, 'no'],
#                [1, 1, 1, 1, 'yes'],
#                [1, 0, 1, 2, 'yes'],
#                [1, 0, 1, 2, 'yes'],
#                [2, 0, 1, 2, 'yes'],
#                [2, 0, 1, 1, 'yes'],
#                [2, 1, 0, 1, 'yes'],
#                [2, 1, 0, 2, 'yes'],
#                [2, 0, 0, 0, 'no']]
# # print(len(dataSet[0]))
# # print(dataSet[0])
# # print(dataSet[1])
# print([example[1] for example in dataSet])
# numFeatures = len(dataSet[0])-1
# # for i in range(numFeatures):
# #     featList = [example[i] for example in dataSet]
# #     uniqueVals = set(featList)
# #     print(uniqueVals)
# #
# # print('\n')
# # print(uniqueVals)
#
# def splitDataSet(dataSet, axis, value):
# 	retDataSet = []										#创建返回的数据集列表
# 	for featVec in dataSet: 							#遍历数据集
# 		if featVec[axis] == value:
# 			reducedFeatVec = featVec[:axis]				#去掉axis特征
# 			reducedFeatVec.extend(featVec[axis+1:]) 	#将符合条件的添加到返回的数据集
# 			retDataSet.append(reducedFeatVec)
# 	return retDataSet
#
# subDataSet = splitDataSet(dataSet,1, 1)
# print(subDataSet)

# from sklearn.neural_network import MLPClassifier
# import numpy as np
#
#
# x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([0, 1, 1, 0])
# mlp_xor = MLPClassifier(hidden_layer_sizes=(20,), activation='relu')
# mlp_xor.fit(x,y)
# y_pred = mlp_xor.predict(x)
# print(y_pred)
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plot
#
# x = np.arange(-10,10,0.02)
# y = np.sin(x)
# y1 = np.sin(x) + 0.2 * np.random.rand(len(x))
# y2 = x**2+3*x+1 + 0.2 * np.random.rand(len(x))
#
# plot.figure(111)
# line1, line2 = plt.plot(x,y,'r', x, y1,'c')
# plt.legend([line1,line2],["r","c"])
# plt.show()

import time
Ptime = time.time()
time.sleep(3)
Ntime = time.time()
print((Ntime - Ptime)/30.0)














import pandas as pd
import numpy as np

# s = pd.Series([1, 2, 88, 4, 5, 6],dtype=int)
# print(s)

mylist = list('abcedfghijklmnopqrstuvwxyz')   # 列表
myarr = np.arange(26)	                      # 数组
mydict = dict(zip(mylist, myarr))             # 字典
# print(mylist)
# print(mydict)
# print(myarr)

ser1 = pd.Series(mydict)
ser2 = pd.Series(mylist)
ser3 = pd.Series(myarr)
print(ser3.head(8))
print(ser3.describe())
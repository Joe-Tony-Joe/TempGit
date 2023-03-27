import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
# import xlwt
import numpy.ma as ma
import re
from scipy.ndimage.interpolation import shift
from matplotlib.animation import FuncAnimation

class ProcessImg():
    def __init__(self):
        self.b_channel,self.g_channel,self.r_channel = [],[],[]
        self.RedVal,self.YelVal = [],[]
        self.H,self.S,self.V = [],[],[]

    def ReadPastImg(self):
        os.chdir(r"F:\下载\NewPic")
        files = os.listdir()
        files.sort()
        for file in files:

            labels = file.split("-")
            if len(file) > 13:
                self.RedVal.append(float(labels[3]))
                self.YelVal.append(float(labels[4]))
            else:
                self.RedVal.append(float(labels[0]))
                self.YelVal.append(float(labels[1]))

            # labels = file.split("-")
            # self.RedVal.append(float(labels[0]))
            # self.YelVal.append(float(labels[1]))

            pics = os.listdir(file)
            img = cv.imread(file + "/" + pics[10])
            b, g, r = cv.split(img)

            # plt.imshow(g,cmap="gray")
            # plt.show()

            b_mean = cv.mean(b)[0]
            g_mean = cv.mean(g)[0]
            r_mean = cv.mean(r)[0]
            self.b_channel.append(b_mean)
            self.r_channel.append(r_mean)
            self.g_channel.append(g_mean)
        os.chdir(r"F:\Code\Pycode\PicDataProcess")

    def ReadImg(self):
        os.chdir(r"F:\下载\NewPic")
        files = os.listdir()
        files.sort()
        # (files)

        for file in files:

            labels = file.split("-")

            # 红光和黄光照射
            if labels[-1] == "Y":
                continue

            #为了和之前写的兼容
            if len(file) > 13:
                self.RedVal.append(float(labels[3]))
                self.YelVal.append(float(labels[4]))
            else:
                self.RedVal.append(float(labels[0]))
                self.YelVal.append(float(labels[1]))

            # labels = file.split("-")
            # self.RedVal.append(float(labels[0]))
            # self.YelVal.append(float(labels[1]))

            pics = os.listdir(file)
            img = cv.imread(file + "/" + pics[10])

            # 这里是对HSV空间处理
            hsv_img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
            h,s,v = cv.split(hsv_img)

            h_mean = cv.mean(h)[0]
            s_mean = cv.mean(s)[0]
            v_mean = cv.mean(v)[0]

            self.H.append(h_mean)
            self.S.append(s_mean)
            self.V.append(v_mean)

            b, g, r = cv.split(img)

            # plt.imshow(g,cmap="gray")
            # plt.show()

            b_mean = cv.mean(b)[0]
            g_mean = cv.mean(g)[0]
            r_mean = cv.mean(r)[0]
            self.b_channel.append(b_mean)
            self.r_channel.append(r_mean)
            self.g_channel.append(g_mean)
        os.chdir(r"F:\PyCode\m_Oil")
        # print(self.r_channel)

    def PloyandPlot(self):
        xh_val = np.linspace(0, 7, 300) # for yellow
        # xh_val = np.linspace(0, 8, 300) # for red
        xv_val = np.linspace(0, 70, 300)

        # H通道判断红值
        phr1 = np.polyfit(self.RedVal, self.YelVal, 1)
        fhr1 = np.poly1d(phr1)
        yhr1 = fhr1(xh_val)
        PreHR = fhr1(self.RedVal)

        print(fhr1)

        # H通道判断黄值
        phy1 = np.polyfit(self.H, self.YelVal, 2)
        fhy1 = np.poly1d(phy1)
        yhy1 = fhy1(xh_val)
        PreHY = fhy1(self.H)
        print(fhy1)

        # V通道红判断红值
        pvr1 = np.polyfit(self.V, self.RedVal, 1)
        fvr1 = np.poly1d(pvr1)
        yvr1 = fvr1(xv_val)
        PreVR = fvr1(self.V)
        print(fvr1)

        # V通道红判断黄值
        pvy1 = np.polyfit(self.V, self.YelVal, 2)
        fvy1 = np.poly1d(pvy1)
        yvy1 = fvy1(xv_val)
        PreVY = fvy1(self.V)
        print(fvy1)
        # yr = (yhr1 + yvr1) / 2
        # yy = (yvr1 + yvy1) / 2


        plt.figure(1)
        plt.subplot(221)
        plt.scatter(self.RedVal, self.YelVal)
        plt.plot(xh_val, yhr1, c="r")
        plt.legend(["Poly","True"])
        plt.xlabel("Redval")
        plt.ylabel("Yelval")

        plt.subplot(222)
        plt.scatter(self.H, self.YelVal)
        plt.plot(xh_val, yhy1, c="r")
        plt.legend(["Poly","True"])
        plt.xlabel("Hval")
        plt.ylabel("YelVal")

        plt.subplot(223)
        plt.scatter(self.V, self.RedVal)
        plt.plot(xv_val, yvr1, c="r")
        plt.legend(["Poly","True"])
        plt.xlabel("Vval")
        plt.ylabel("RedVal")

        plt.subplot(224)
        plt.scatter(self.V, self.YelVal)
        plt.plot(xv_val, yvy1, c="r")
        plt.legend(["Poly","True"])
        plt.xlabel("Vval")
        plt.ylabel("YelVal")



        plt.figure(2)

        h_val = np.linspace(0,len(self.H),len(self.H))
        v_val = np.linspace(0,len(self.V),len(self.V))

        # h_val,v_val = [],[]
        h_val = np.array(h_val)
        v_val = np.array(v_val)
        PreHR = np.array(PreHR)
        PreVR = np.array(PreVR)
        PreHY = np.array(PreHY)
        PreVY = np.array(PreVY)
        RedVal = np.array(self.RedVal)
        YelVal = np.array(self.YelVal)

        plt.subplot(221)
        plt.stem(h_val, PreHR - RedVal)
        plt.xlabel("h_val")
        plt.title("meanBias:{:.2}".format(np.mean(abs(PreHR - RedVal))))
        plt.ylim([-1, 1])

        plt.subplot(222)
        plt.stem(v_val, PreVR - RedVal)
        plt.xlabel("v_val")
        plt.title("meanBias:{:.2}".format(np.mean(abs(PreVR - RedVal))))
        plt.ylim([-1, 1])

        plt.subplot(223)
        plt.stem(h_val, PreHY - YelVal)
        plt.xlabel("h_val")
        plt.title("meanBias:{:.2}".format(np.mean(abs(PreHY - YelVal))))
        plt.ylim([-25, 25])

        plt.subplot(224)
        plt.stem(v_val, PreVY - YelVal)
        plt.xlabel("v_val")
        plt.title("meanBias:{:.2}".format(np.mean(abs(PreVY - YelVal))))
        plt.ylim([-25, 25])
        # plt.subplot_tool()
        plt.subplots_adjust(left=0.125,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.2,
                            hspace=0.35)

        

        HDataVal = np.vstack([h_val, YelVal, PreHY])
        VDataVal = np.vstack([v_val, YelVal, PreVY])
        HDataVal = np.sort(HDataVal)
        VDataVal = np.sort(VDataVal)
        HDataVal = HDataVal[:, :-4]
        VDataVal = VDataVal[:, :-4]
        plt.figure(3)
        plt.subplot(121)
        plt.stem(HDataVal[0], HDataVal[2] - HDataVal[1])
        plt.title("meanBias:{:.2}".format(np.mean(abs(HDataVal[2] - HDataVal[1]))))

        plt.subplot(122)
        plt.stem(VDataVal[0], VDataVal[2] - VDataVal[1])
        plt.stem(VDataVal[0], VDataVal[2] - VDataVal[1])
        plt.title("meanBias:{:.2}".format(np.mean(abs(VDataVal[2] - VDataVal[1]))))

        # plt.subplot_tool()
        plt.show()


    def PloyandPlot2(self):
        xr_val = np.linspace(0, 7, 300)
        xg_val = np.linspace(0, 70, 300)

        prr1 = np.polyfit(self.RedVal, self.Yelval, 1)
        frr1 = np.poly1d(prr1)
        yrr1 = frr1(xr_val)
        PreRR = frr1(self.RedVal)

        print(frr1)

        # R通道红判断黄值
        pry1 = np.polyfit(self.r_channel, self.YelVal, 2)
        fry1 = np.poly1d(pry1)
        yry1 = fry1(xr_val)
        PreRY = fry1(self.r_channel)
        print(fry1)

        # G通道红判断红值
        pgr1 = np.polyfit(self.g_channel, self.RedVal, 1)
        fgr1 = np.poly1d(pgr1)
        ygr1 = fgr1(xg_val)
        PreGR = fgr1(self.g_channel)
        print(fgr1)

        # G通道红判断黄值
        pgy1 = np.polyfit(self.g_channel, self.YelVal, 2)
        fgy1 = np.poly1d(pgy1)
        ygy1 = fgy1(xg_val)
        PreGY = fgy1(self.g_channel)
        print(fgy1)
        # yr = (yrr1 + ygr1) / 2
        # yy = (yry1 + ygy1) / 2

        plt.figure(1)
        plt.subplot(221)
        plt.scatter(self.Redval, self.Yelval)
        plt.plot(xr_val, yrr1, c="r")
        plt.legend(["Poly","True"])
        plt.xlabel("Rval")
        plt.ylabel("Yval")

        plt.subplot(222)
        plt.scatter(self.r_channel, self.YelVal)
        plt.plot(xr_val, yry1, c="r")
        plt.legend(["Poly","True"])
        plt.xlabel("r_val")
        plt.ylabel("YelVal")

        plt.subplot(223)
        plt.scatter(self.g_channel, self.RedVal)
        plt.plot(xg_val, ygr1, c="r")
        plt.legend(["Poly","True"])
        plt.xlabel("g_val")
        plt.ylabel("RedVal")

        plt.subplot(224)
        plt.scatter(self.g_channel, self.YelVal)
        plt.plot(xg_val, ygy1, c="r")
        plt.legend(["Poly","True"])
        plt.xlabel("g_val")
        plt.ylabel("YelVal")

        plt.figure(2)

        r_val = np.linspace(0,len(self.r_channel),len(self.r_channel))
        g_val = np.linspace(0,len(self.g_channel),len(self.g_channel))

        # r_val,g_val = [],[]
        r_val = np.array(r_val)
        g_val = np.array(g_val)
        PreRR = np.array(PreRR)
        PreGR = np.array(PreGR)
        PreRY = np.array(PreRY)
        PreGY = np.array(PreGY)
        RedVal = np.array(self.RedVal)
        YelVal = np.array(self.YelVal)

        plt.subplot(221)
        plt.stem(r_val, PreRR - RedVal)
        plt.title("meanBias:{:.2}".format(np.mean(abs(PreRR - RedVal))))
        plt.ylim([-1, 1])

        plt.subplot(222)
        plt.stem(g_val, PreGR - RedVal)
        plt.title("meanBias:{:.2}".format(np.mean(abs(PreGR - RedVal))))
        plt.ylim([-1, 1])

        plt.subplot(223)
        plt.stem(r_val, PreRY - YelVal)
        plt.title("meanBias:{:.2}".format(np.mean(abs(PreRY - YelVal))))
        plt.ylim([-25, 25])

        plt.subplot(224)
        plt.stem(g_val, PreGY - YelVal)
        plt.title("meanBias:{:.2}".format(np.mean(abs(PreGY - YelVal))))
        plt.ylim([-25, 25])

        RDataVal = np.vstack([r_val, YelVal, PreRY])
        GDataVal = np.vstack([g_val, YelVal, PreGY])
        RDataVal = np.sort(RDataVal)
        GDataVal = np.sort(GDataVal)
        RDataVal = RDataVal[:, :-4]
        GDataVal = GDataVal[:, :-4]
        plt.figure(3)
        plt.subplot(121)
        plt.stem(RDataVal[0], RDataVal[2] - RDataVal[1])
        plt.title("meanBias:{:.2}".format(np.mean(abs(RDataVal[2] - RDataVal[1]))))

        plt.subplot(122)
        plt.stem(GDataVal[0], GDataVal[2] - GDataVal[1])
        plt.stem(GDataVal[0], GDataVal[2] - GDataVal[1])
        plt.title("meanBias:{:.2}".format(np.mean(abs(GDataVal[2] - GDataVal[1]))))

        plt.show()


    def Denoise(self):
        os.chdir(r"E:\Picdata\5900-8-3")
        # os.chdir(r"F:\Picdata\2500")
        files = os.listdir()
        files.sort(key=lambda file: float(re.split("-", file)[0]))
        for file in files:
            labels = file.split("-")
            if len(file)>13:
                self.RedVal.append(float(labels[3]))
                self.YelVal.append(float(labels[4]))
            else:
                self.RedVal.append(float(labels[0]))
                self.YelVal.append(float(labels[1]))

            # print(float(labels[0]),float(labels[1]))


            pics = os.listdir(file)
            img = cv.imread(file + "/" + pics[10])
            b, g, r = cv.split(img)

            b_mean = cv.mean(b)[0]

            #cv的去噪，基本没啥效果
            # Gcopy = cv.fastNlMeansDenoising(g)
            # Rcopy = cv.fastNlMeansDenoising(r)
            # g_mean = cv.mean(Gcopy)[0]
            # r_mean = cv.mean(Rcopy)[0]

            #没用
            # b_numpy = np.asarray(b,np.uint8)
            # modifyIDB = np.where(b_numpy<0.8*b_mean)
            # # print(np.where(b_numpy<0.8*b_mean))p
            # for rowb,colb in zip(modifyIDB[0],modifyIDB[1]):
            #     b[rowb][colb] = b_mean

            # numpy查找实现降噪，没有Sobel合理
            # Gcopy = img.copy()
            # Rcopy = img.copy()
            #
            # g_mean = cv.mean(g)[0]
            # g_numpy = np.asarray(g, np.uint8)
            # modifyIDG = np.where(g_numpy <0.9* g_mean)
            # mask = np.zeros(g_numpy.shape)
            # for rowg, colg in zip(modifyIDG[0], modifyIDG[1]):
            #     # g[rowg][colg] = g_mean
            #     mask[rowg][colg] = 1
            #     Gcopy[rowg][colg][:] = np.asarray([0, 255, 0])
            # g_numpy = ma.array(g_numpy,mask=mask)
            # g_mean = np.mean(g_numpy)
            #
            #
            # r_mean = cv.mean(r)[0]
            # r_numpy = np.asarray(r, np.uint8)
            # mask = np.zeros(r_numpy.shape)
            # modifyIDR = np.where(r_numpy < 0.9*r_mean)
            # for rowr, colr in zip(modifyIDR[0], modifyIDR[1]):
            #     mask[rowr][colr] = 1
            #     Rcopy[rowr][colr][:] = np.asarray([0, 0, 255])
            # r_numpy = ma.array(r_numpy,mask=mask)
            # r_mean = np.mean(r_numpy)

            # print("g_mean", g_mean)
            # print("r_mean", r_mean)


            # 做背景减法的
            # grMASK = np.zeros(g.shape,np.uint8)
            # cv.subtract(r,g,grMASK)

            # Sobel算子解
            Gcopy = img.copy()
            Rcopy = img.copy()

            sobelGXY = cv.Sobel(g,cv.CV_64F,1,1,ksize=3)

            # LaplacianGXY = cv.Laplacian(g, cv.CV_64F, ksize=3)

            # sobelGXY = np.asarray(sobelGXY,dtype=np.float)

            # Gmin = np.argmin(sobelGXY) #不能用，数据转换错误


            # min_val,max_val,min_indx,max_indx = cv.minMaxLoc(sobelGXY)
            Gmin,Gmax,_,_ = cv.minMaxLoc(sobelGXY)
            # print("G:",Gmin,Gmax)


            maskG = np.zeros(g.shape,np.uint8)
            GmodifyID = np.where((sobelGXY<0.3*Gmin)|(sobelGXY>0.7*Gmax))

            for row,col in zip(GmodifyID[0],GmodifyID[1]):
                maskG[row][col] = 1
                # if row>3 and col >3:
                #     if(maskG[row-3][col-3]==1):
                #         for i in range(row-3,row):
                #             for j in range(col-3,col):
                #                 maskG[i][j]=1
                #     if (maskG[row - 2][col - 2] == 1):
                #         for i in range(row - 2, row):
                #             for j in range(col - 2, col):
                #                 maskG[i][j] = 1
                #     if (maskG[row - 1][col - 1] == 1):
                #         for i in range(row - 1, row):
                #             for j in range(col - 1, col):
                #                 maskG[i][j] = 1
                Gcopy[row][col][:] = np.asarray([0, 255, 0])
            # kernel = np.ones((3,3))
            # maskG = cv.morphologyEx(maskG,cv.MORPH_CLOSE,kernel)
            # maskG1 = shift(maskG,[3,3],cval=0)
            # maskG2 = shift(maskG, [-3, -3], cval=0)
            # maskG = np.logical_and(maskG1,maskG2)

            NewGID = np.where(maskG==1)
            for row,col in zip(NewGID[0],NewGID[1]):
                Gcopy[row][col][:] = np.asarray([0,255,0])
                # g[row][col] = 255

            g_numpy = ma.array(g,mask=maskG)
            g_mean = np.mean(g_numpy)

            sobelRXY = cv.Sobel(r, cv.CV_64F, 1, 1, ksize=3)
            # LaplacianRXY = cv.Laplacian(r, cv.CV_64F, ksize=3)

            Rmin, Rmax, _, _ = cv.minMaxLoc(sobelRXY)
            # print("R:", Rmin, Rmax)


            maskR = np.zeros(r.shape,np.uint8)
            RmodifyID = np.where((sobelRXY < 0.3*Rmin)|(sobelRXY>0.7*Gmax))
            for row, col in zip(RmodifyID[0], RmodifyID[1]):
                maskR[row][col] = 1
                # if row>3 and col >3:
                #     if(maskR[row-3][col-3]==1):
                #         for i in range(row - 3, row):
                #             for j in range(col - 3, col):
                #                 maskR[i][j] = 1
                #     if (maskR[row - 2][col - 2] == 1):
                #         for i in range(row - 2, row):
                #             for j in range(col - 2, col):
                #                 maskR[i][j] = 1
                #     if (maskR[row - 1][col - 1] == 1):
                #         for i in range(row - 1, row):
                #             for j in range(col - 1, col):
                #                 maskR[i][j] = 1
                Rcopy[row][col][:] = np.asarray([0, 0, 255])
            # maskR = cv.morphologyEx(maskR, cv.MORPH_CLOSE, kernel)
            # maskR1 = shift(maskG, [3, 3], cval=0)
            # maskR2 = shift(maskG, [-3, -3], cval=0)
            # maskR = np.logical_and(maskR1,maskR2)
            # NewRID = np.where(maskR == 1)
            # for row, col in zip(NewRID[0], NewRID[1]):
            #     Rcopy[row][col][:] = np.asarray([0, 0, 255])
            #     r[row][col] = 255


            r_numpy = ma.array(r, mask=maskR)
            r_mean = np.mean(r_numpy)







            self.b_channel.append(b_mean)
            self.r_channel.append(r_mean)
            self.g_channel.append(g_mean)


            #
            plt.subplot(231)
            plt.imshow(g,cmap="gray")
            # plt.titie("GChannel")

            plt.subplot(232)
            plt.imshow(r,cmap="gray")
            # plt.title("RChannel")

            plt.subplot(233)
            img_plt = cv.cvtColor(img,cv.COLOR_BGR2RGB)
            plt.imshow(img_plt)

            plt.subplot(234)
            Gcopy = cv.cvtColor(Gcopy,cv.COLOR_BGR2RGB)
            plt.imshow(Gcopy)

            plt.subplot(235)
            Rcopy = cv.cvtColor(Rcopy, cv.COLOR_BGR2RGB)
            plt.imshow(Rcopy)

            plt.show()

        os.chdir(r"F:\Code\Pycode\PicDataProcess")

    def Fun(self,x):
        return - (9603 * x ** 4) / 10000 + (418 * x ** 3) / 25 - (522 * x ** 2) / 5 + (537 * x) / 2 - 1823 / 10


    def Red2Yel(self):

        x = np.linspace(1,7,300)
        y = self.Fun(x)

        plt.figure(1)
        plt.scatter(self.RedVal,self.YelVal)
        plt.plot(x,y,c = "r")
        plt.show()


    def JustPlot(self):
        # plt.figure(1)
        #
        # plt.subplot(221)
        # plt.scatter(self.r_channel, self.RedVal, c="r")
        # plt.xlabel("r_channel")
        # plt.ylabel("RedVal")
        # plt.subplot(222)
        # plt.scatter(self.r_channel, self.YelVal, c="b")
        # plt.xlabel("r_channel")
        # plt.ylabel("YelVal")
        # plt.subplot(223)
        # plt.scatter(self.g_channel, self.RedVal, c="r")
        # plt.xlabel("g_channel")
        # plt.ylabel("RedVal")
        # plt.subplot(224)
        # plt.scatter(self.g_channel, self.YelVal, c="b")
        # plt.xlabel("g_channel")
        # plt.ylabel("YelVal")
        # plt.show()


        plt.figure(2)
        plt.subplot(321)
        plt.scatter(self.H, self.RedVal, c="r")
        plt.xlabel("H")
        plt.ylabel("RedVal")
        plt.subplot(322)
        plt.scatter(self.H, self.YelVal, c="b")
        plt.xlabel("H")
        plt.ylabel("YelVal")

        plt.subplot(323)
        plt.scatter(self.S, self.RedVal, c="r")
        plt.xlabel("S")
        plt.ylabel("RedVal")
        plt.subplot(324)
        plt.scatter(self.S, self.YelVal, c="b")
        plt.xlabel("S")
        plt.ylabel("YelVal")

        plt.subplot(325)
        plt.scatter(self.V, self.RedVal, c="r")
        plt.xlabel("V")
        plt.ylabel("RedVal")
        plt.subplot(326)
        plt.scatter(self.V, self.YelVal, c="r")
        plt.xlabel("V")
        plt.ylabel("YelVal")


        plt.show()

    def Aniation(self):

        # fig = plt.figure(figsize=(10,5))
        #
        plt.rcParams["font.family"] = "Adobe Heiti Std"
        # x = []
        # y = []
        # plt.xlim(0,30)
        # plt.ylim(0,6)
        # plt.title("黄光照射") # 红光照射
        # plt.xlabel("Hval")
        # plt.ylabel("RedVal")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(20,9))

        x1 ,y1 = [],[]
        ax1.set_xlim([0,30])
        ax1.set_ylim([0, 6])
        ax1.set_title("黄光照射")  # 红光照射
        ax1.set_xlabel("Hval")
        ax1.set_ylabel("RedVal")

        x2,y2 = [],[]
        ax2.set_xlim([200,255])
        ax2.set_ylim([0, 6])
        ax2.set_title("黄光照射")  # 红光照射
        ax2.set_xlabel("Vval")
        ax2.set_ylabel("RedVal")

        x3, y3= [], []
        ax3.set_xlim([0,30])
        ax3.set_ylim([0, 70])
        ax3.set_title("黄光照射")  # 红光照射
        ax3.set_xlabel("Hval")
        ax3.set_ylabel("YelVal")

        x4, y4 = [], []
        ax4.set_xlim([200,255])
        ax4.set_ylim([0, 70])
        ax4.set_title("黄光照射")  # 红光照射
        ax4.set_xlabel("Vval")
        ax4.set_ylabel("YelVal")

        # x1 ,y1 = [],[]
        # ax1.set_ylim([0,30])
        # ax1.set_xlim([0, 6])
        # ax1.set_title("黄光照射")  # 红光照射
        # ax1.set_ylabel("Hval")
        # ax1.set_xlabel("RedVal")
        #
        # x2,y2 = [],[]
        # ax2.set_ylim([200,255])
        # ax2.set_xlim([0, 6])
        # ax2.set_title("黄光照射")  # 红光照射
        # ax2.set_ylabel("Vval")
        # ax2.set_xlabel("RedVal")
        #
        # x3, y3= [], []
        # ax3.set_ylim([0,30])
        # ax3.set_xlim([0, 70])
        # ax3.set_title("黄光照射")  # 红光照射
        # ax3.set_ylabel("Hval")
        # ax3.set_xlabel("YelVal")
        #
        # x4, y4 = [], []
        # ax4.set_ylim([200,255])
        # ax4.set_xlim([0, 70])
        # ax4.set_title("黄光照射")  # 红光照射
        # ax4.set_ylabel("Vval")
        # ax4.set_xlabel("YelVal")

        # x1, y1 = [], []
        # ax1.set_xlim([0, 8])
        # ax1.set_ylim([0, 6])
        # ax1.set_title("红光照射")
        # ax1.set_xlabel("Hval")
        # ax1.set_ylabel("RedVal")
        #
        # x2, y2 = [], []
        # ax2.set_xlim([200, 255])
        # ax2.set_ylim([0, 6])
        # ax2.set_title("红光照射")
        # ax2.set_xlabel("Vval")
        # ax2.set_ylabel("RedVal")
        #
        # x3, y3 = [], []
        # ax3.set_xlim([0, 8])
        # ax3.set_ylim([0, 70])
        # ax3.set_title("红光照射")
        # ax3.set_xlabel("Hval")
        # ax3.set_ylabel("YelVal")
        #
        # x4, y4 = [], []
        # ax4.set_xlim([200, 255])
        # ax4.set_ylim([0, 70])
        # ax4.set_title("红光照射")
        # ax4.set_xlabel("Vval")
        # ax4.set_ylabel("YelVal")



        def update1(n):
            x1.append(self.H[n])
            y1.append(self.RedVal[n])

            ax1.plot(x1,y1,color = 'green',ls = '--',marker = 'o',alpha=0.3)

        def update2(n):
            x2.append(self.V[n])
            y2.append(self.RedVal[n])

            ax2.plot(x2, y2, color = 'cyan',ls = '--',marker = 'o',alpha=0.3)

        def update3(n):
            x3.append(self.H[n])
            y3.append(self.YelVal[n])

            ax3.plot(x3, y3, color = 'skyblue',ls = '--',marker = 'o',alpha=0.3)

        def update4(n):
            x4.append(self.V[n])
            y4.append(self.YelVal[n])

            ax4.plot(x4, y4, color = 'pink',ls = '--',marker = 'o',alpha=0.3)

        ani1 = FuncAnimation(fig,update1,frames=np.arange(0,len(self.RedVal),1),interval=450,blit=False,
                             repeat=False)
        ani2 = FuncAnimation(fig, update2, frames=np.arange(0, len(self.RedVal), 1), interval=450, blit=False,
                             repeat=False)
        ani3 = FuncAnimation(fig, update3, frames=np.arange(0, len(self.RedVal), 1), interval=450, blit=False,
                             repeat=False)
        ani4 = FuncAnimation(fig, update4, frames=np.arange(0, len(self.RedVal), 1), interval=450, blit=False,
                             repeat=False)

        plt.show()

    def Aniation2(self):

        plt.rcParams["font.family"] = "Adobe Heiti Std"

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 9))

        x1, y1 = [], []
        ax1.set_xlim([0, 260])
        ax1.set_ylim([0, 6])
        # ax1.set_title("黄光照射")
        ax1.set_xlabel("r_channel")
        ax1.set_ylabel("RedVal")

        x2, y2 = [], []
        ax2.set_xlim([0, 260])
        ax2.set_ylim([0, 6])
        # ax2.set_title("黄光照射")
        ax2.set_xlabel("g_channel")
        ax2.set_ylabel("RedVal")

        x3, y3 = [], []
        ax3.set_xlim([0, 260])
        ax3.set_ylim([0, 70])
        # ax3.set_title("黄光照射")
        ax3.set_xlabel("r_channel")
        ax3.set_ylabel("YelVal")

        x4, y4 = [], []
        ax4.set_xlim([0, 260])
        ax4.set_ylim([0, 70])
        # ax4.set_title("黄光照射")
        ax4.set_xlabel("g_channel")
        ax4.set_ylabel("YelVal")


        def update1(n):
            x1.append(self.r_channel[n])
            y1.append(self.RedVal[n])

            ax1.plot(x1, y1, color='green', ls='--', marker='o', alpha=0.3)

        def update2(n):
            x2.append(self.g_channel[n])
            y2.append(self.RedVal[n])

            ax2.plot(x2, y2, color='cyan', ls='--', marker='o', alpha=0.3)

        def update3(n):
            x3.append(self.r_channel[n])
            y3.append(self.YelVal[n])

            ax3.plot(x3, y3, color='skyblue', ls='--', marker='o', alpha=0.3)

        def update4(n):
            x4.append(self.g_channel[n])
            y4.append(self.YelVal[n])

            ax4.plot(x4, y4, color='pink', ls='--', marker='o', alpha=0.3)

        ani1 = FuncAnimation(fig, update1, frames=np.arange(0, len(self.RedVal), 1), interval=450, blit=False,
                             repeat=False)
        ani2 = FuncAnimation(fig, update2, frames=np.arange(0, len(self.RedVal), 1), interval=450, blit=False,
                             repeat=False)
        ani3 = FuncAnimation(fig, update3, frames=np.arange(0, len(self.RedVal), 1), interval=450, blit=False,
                             repeat=False)
        ani4 = FuncAnimation(fig, update4, frames=np.arange(0, len(self.RedVal), 1), interval=450, blit=False,
                             repeat=False)

        plt.show()








if __name__=="__main__":
    IMGs = ProcessImg()
    # IMGs.ReadImg()
    # IMGs.Aniation()


    IMGs.ReadImg()
    IMGs.PloyandPlot()
    # IMGs.Aniation()


    # IMGs.ReadPastImg()
    # IMGs.PloyandPlot2()












    # from matplotlib import font_manager
    #
    # a = sorted([f.name for f in font_manager.fontManager.ttflist])
    # for i in a:
    #     print(i)




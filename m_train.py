import os
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from PIL import Image



class OilImage(Dataset):
    def __init__(self,root,train = True,transform = None,target_transform=None):
        super(OilImage, self).__init__()
        self.train = train
        self.labels = []
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            img_folders = root+"/train"

        else:
            img_folders = root+"/test"

        self.img_list=[]
        for dir in os.listdir(img_folders): #dir :0-0
            pics = os.listdir(img_folders+"/"+dir)
            for i in range(100):
                self.img_list.append(img_folders+"/"+dir+"/"+pics[i])
                label = str.split(dir,"-")
                label = np.array(label, dtype=float)
                self.labels.append(label)



    def __getitem__(self, index):
        # img_name = self.img_folder + self.filenames[index]
        # print(index)
        img_name = self.img_list[index]

        img = Image.open(img_name)
        img = self.transform(img)  # 可以根据指定的转化形式对数据集进行转换

        label = self.labels[index]
        label = torch.Tensor(label)

        # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
        return img, label

    def __len__(self):
        # print(len(self.labels))
        return len(self.labels)






def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(7 * 7 * 512, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = nn.Linear(4096, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out[:,1] = 30*out[:,1]

        return vgg16_features, out

root = "Picture"

data_transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

EPOCH = 1000              # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
LR = 0.001

train_dataset = OilImage(root,train=True,transform=data_transform)
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)

test_dataset = OilImage(root,train=True,transform=data_transform)
test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)

model = VGG16()
model.cuda()


loss_fun = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

for epoch in range(EPOCH):
    avg_loss = 0
    cnt = 0
    for step,(img,label) in enumerate(train_loader,0):
        img, label = Variable(img),Variable(label)
        images = img.cuda()
        labels = label.cuda()

        optimizer.zero_grad()
        _,output = model(images)

        if(epoch%10==0):
            print("------------Output--------------")
            print(output)
            print("------------TrueLabel-----------")
            print(labels)

        loss = loss_fun(output,labels)

        avg_loss += loss.data
        cnt += 1
        print("[E: %d] loss: %f, avg_loss: %f" % (epoch, loss.data, avg_loss / cnt))
        loss.backward()
        optimizer.step()

    scheduler.step(avg_loss)
    if epoch>500 and epoch%100 == 0:
        torch.save(model, 'net'+str(epoch)+'.pkl')






















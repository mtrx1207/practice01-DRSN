from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt

from rsnet import rsnet34
from data_process import preprocess

path = r"D:\ZJU\coding scratch\python\DRSN-1D\test_data\data_snr-5"
data_mark = "DE"
fs = 12000
win_tlen = 2048 / 12000
overlap_rate = (0 / 2048) * 100
random_seed = 1
batch_size=16
num_epochs=10
X, y = preprocess(path,
                  data_mark,
                  fs, # fs * win_tlen = win_len
                  win_tlen, #fs * win_tlen = win_len
                  overlap_rate,
                  random_seed
                  )
len_data=len(X)
print(len_data)
X=X.reshape(len_data,1,2048,1)
train_data = torch.from_numpy(X) #convert numpy to tensor, dtype float64
train_label = torch.from_numpy(y) #convert numpy to tensor, dtype int32
#train_data = type(torch.FloatTensor)
#train_label = type(torch.FloatTensor)
train_dataset = TensorDataset(train_data, train_label)

train_size = int(len(train_dataset) * 0.7)
test_size = len(train_dataset) - train_size

#split into train dataset (70%) and test dataset (30%) RANDOMLY
#train_dataset and test_dataset is a torch.utils.data.Subset, it's also a part of Dataset
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

# put train and test Dataset into DataLoader
# set batch_size = 16
# shuffle = True, reshuffle every data on different epoch
# drop_last = True, drop the last incomplete batch (occur when data size is indivisible by batch size)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
test_data_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

# put our rsnet model into CPU
net = rsnet34()
net = net.cpu()

Loss_list = []
Accuracy_list = []
acc = []

#use Stochastic Gradient Descent optimization algorithm
# params: iterable of parameters to optimize from our model
# lr: hyperparam, learning rate default = 1e-3, adjust training speed(lr * step_size). too large cause instability (converge too fast), too small cause learning process too slow
# momentum: hyperparam, change (increase) lr in each epochs when the error cost gradient is heading in the same direction for a long time and avoid local minima by rolling over small bumps
# weight_decay: hyperparam, change (reduce) lr in each epochs when a too high lr makes the learning jump back and forth over a minimum penalty 

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# measuring classification model whose output is probability in range (0,1).
# example: prediction output = 0.012, actual probability = 1. this will result in huge error
loss_function = nn.CrossEntropyLoss()

#epochs starts here, each one epoch includes training and testing phase
for epoch in range(num_epochs):
    net.train() #set our model in training mode (activate Dropout and BatchNorm)
    sum_loss = 0.0  # 损失数量
    correct = 0.0  # 准确数量
    total = 0.0  # 总共数量
    
    #training loop starts here
    for i,(X,y) in enumerate(train_data_loader):
        length = len(train_data_loader)
        X = X.type(torch.FloatTensor) #change dtype from torch.float64 (numpy default) to torch.float32
        y = y.type(torch.LongTensor) #change dtype from torch.int32 to torch.int64
        #y = y.type(torch.cuda.FloatTensor)

        optimizer.zero_grad() #reset the gradients of all optimized tensors
        outputs = net(X) #CALL THE FORWARD METHOD INSIDE OUR MODEL, return a tensor
        loss = loss_function(outputs, y) #compare output and label, return a tensor
        loss.backward() #computes gradient of current tensor
        optimizer.step() #updates all parameters in our model

        sum_loss += loss.item() # keep track total loss in each epoch
        _, predicted = torch.max(outputs.data, 1) #return a tuple (max values, corresponding indices) from outputs.data, 1 is dim
        total += y.size(0) # total data length
        correct += predicted.eq(y.data).cpu().sum() #compute element wise equality
        print('[epoch:%d, iter:%d/%d] Loss: %.03f | Acc: %.3f%% '
              % (epoch + 1, (i + 1), length, sum_loss / (i + 1), 100. * correct / total))
    Loss_list.append(sum_loss / (len(train_data_loader))) #average loss in each epoch, for visualization
    Accuracy_list.append(correct / total) #average accuracy in each epoch, for visualization

    #testing/evaluating phase starts here
    print("Waiting Test!")
    with torch.no_grad():  # 没有求导
        correct = 0
        total = 0
        for test_i,(test_X,test_y) in enumerate(test_data_loader):
            net.eval()  # 运用net.eval()时，由于网络已经训练完毕，参数都是固定的，因此每个min-batch的均值和方差都是不变的，因此直接运用所有batch的均值和方差。
            X = test_X.type(torch.FloatTensor)
            y = test_y.type(torch.LongTensor)
            outputs = net(X)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1) 
            total += y.size(0) 
            correct += (predicted == y).sum().item()
            if test_i == 100:
                break
        print('测试分类准确率为：{}%'.format(round(100 * correct / total, 3)))
        acc.append( 100. * correct / total)

# plot graphs of accuracy and loss progression during each epoch
x1 = range(0, num_epochs)
x2 = range(0, num_epochs)
x3 = range(0, num_epochs)
y1 = Accuracy_list
y2 = Loss_list
y3 = acc
plt.subplot(3, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Train accuracy vs. epoches')
plt.ylabel('Train accuracy')
plt.subplot(3, 1, 2)
plt.plot(x2, y2, '.-')
plt.title('Train loss vs. epoches')
plt.ylabel('Train loss')
plt.subplot(3,1,3)
plt.plot(x3, y3, 'o-')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.show()
#print(Accuracy_list)
print("Training Finished, TotalEPOCH=%d" % num_epochs)


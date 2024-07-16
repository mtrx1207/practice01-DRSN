"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import torchsummary


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.shrinkage = Shrinkage(out_channels, gap_size=(1, 1))
        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), stride=stride, padding=(0,1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=(1,3), padding=(0,1), bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            self.shrinkage
        )
        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(gap_size) #2d average pooling
        
        # Modules will be added to it in the order they are passed in the constructor.
        self.fc = nn.Sequential(
            #linear transformation, y = xA^t + b 
            # 2 arguments: in_features (size of each input sample), out_features (size of each output sample)
            nn.Linear(channel, channel), 
            
            #apply batch normalization, batch size = channel
            #formula on documentation https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
            nn.BatchNorm1d(channel),
            
            #ReLU(x) = max(0,x)
            #linear activation function
            #mitigating the risk of vanishing gradient (no gradient, no learning)
            #for example, sigmoid function squeeze the data and during backpropagation the gradient is getting smaller and smaller
            nn.ReLU(inplace=True),
            
            #another linear transformation
            nn.Linear(channel, channel),
            
            #Sigmoid(x) = 1/(1+e^-x) activation function
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x) #convert all data to positive
        x_abs = x
        x = self.gap(x) #2d pooling
        x = torch.flatten(x, 1) #flattening the tensor, start_dim = 1
        # average = torch.mean(x, dim=1, keepdim=True)
        average = x
        x = self.fc(x) #feed the data to our sequential model
        x = torch.mul(average, x) #element wise multiplication
        x = x.unsqueeze(2).unsqueeze(2) 
        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros) #remove negative, kind of ReLU? is this the heart of residual model?
        #torch sign return an output torch = sgn(input torch)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


class RSNet(nn.Module):

    def __init__(self, block, num_block, num_classes=4):
        super().__init__()

        self.in_channels = 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=[1,3], padding=(0,1), bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 4, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 8, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 16, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 32, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make rsnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual shrinkage block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a rsnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def rsnet18():
    """ return a RsNet 18 object
    """
    return RSNet(BasicBlock, [2, 2, 2, 2])

def rsnet34():
    """ return a RsNet 34 object
    """
    return RSNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return RsNet(BottleNeck, [3, 4, 6, 3])


model=rsnet34()
model=model.cpu()

from torchsummary import summary
summary(model,input_size=(1,2048,1),batch_size=-1,device='cpu')


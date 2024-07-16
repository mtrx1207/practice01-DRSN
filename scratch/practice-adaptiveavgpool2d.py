import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
print(torch.cuda.is_available())

img = torch.arange(28, dtype=torch.float32).reshape(1,4,7)

""" 
case 1(divisible size):
    [0 , 1 , 2 , 3 , 4 , 5 ]
    [6 , 7 , 8 , 9 , 10, 11]
    [12, 13, 14, 15, 16, 17]
    [18, 19, 20, 21, 22, 23]

    count avgpool2d(2,3):
    [0 , 1]   [2 , 3]   [4 , 5]
    [6 , 7]   [8 , 9]   [10,11]

    [12,13]   [14,15]   [16,17]
    [18,19]   [20,21]   [22,23]

case 2 (undivisible size):
    [0 , 1 , 2 , 3 , 4 , 5 , 6 ]
    [7 , 8 , 9 , 10, 11, 12, 13]
    [14, 15, 16, 17, 18, 19, 20]
    [21, 22, 23, 24, 25, 26, 27]
    
    count avgpool2d(2,3), more like sliding window with overlap:
    [0,1,2]      [2,3,4]     [4,5,6] <-- 3rd and 6th column are overlapped
    [7,8,9]      [9,10,11]   [11,12,13]

    [14,15,16]   [16,17,18]  [18,19,20]
    [21,22,23]   [23,24,25]  [25,26,27]

"""

#input dimension should be at least 3! If it doesn't, just unsqueeze
pool_1 = nn.AdaptiveAvgPool2d((2,3))
pool_2 = nn.AdaptiveAvgPool2d(2)
img_1 = pool_1(img)
img_2 = pool_2(img)
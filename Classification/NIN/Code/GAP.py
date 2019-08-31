import torch
from torch import nn

a = torch.arange(36.).reshape(2, 2, 3, 3)
print(a)
'''
tensor([[[[ 0.,  1.,  2.],
          [ 3.,  4.,  5.],
          [ 6.,  7.,  8.]],

         [[ 9., 10., 11.],
          [12., 13., 14.],
          [15., 16., 17.]]],


        [[[18., 19., 20.],
          [21., 22., 23.],
          [24., 25., 26.]],

         [[27., 28., 29.],
          [30., 31., 32.],
          [33., 34., 35.]]]])
'''

gap = nn.AvgPool2d(3)
print(gap(a))
# tensor([[[[ 4.]],

#          [[13.]]],


#         [[[22.]],

#          [[31.]]]])

print(gap(a).shape)
# torch.Size([2, 2, 1, 1])

print(gap(a).reshape(2, 2))
print(gap(a).view(2, 2))
# tensor([[ 4., 13.],
#         [22., 31.]])

import torch
import torch.nn.functional as F
from basic_conv2d import BasicConv2d

class InceptionA(torch.nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()


        self.pool_features = 64
        self.branch1x1_size  = 64
        self.branch5x5_size  = 64
        self.branch3x3_size  = 64

        self.branch1x1 = BasicConv2d(in_channels, self.branch1x1_size, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, self.branch5x5_size, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, self.branch3x3_size, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, self.pool_features, kernel_size=1)

    def output_feature_shape(self):
        return self.pool_features + self.branch1x1_size + self.branch3x3_size + self.branch5x5_size

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)



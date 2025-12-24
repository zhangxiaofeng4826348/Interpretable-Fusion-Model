import torch
import torch.nn as nn
from torch.nn import functional as F

# 基本卷积块
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.3),
            nn.ReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.4),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)


# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 3, 2, 1),
            nn.BatchNorm2d(C),
            nn.ReLU()
        )

    def forward(self, x):
        return self.Down(x)

# 上采样模块
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用双线性插值进行上采样
        up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)

class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        # 4次下采样
        self.C1 = Conv(3, 32)
        self.D1 = DownSampling(32)
        self.global_max_pool1 = nn.AdaptiveMaxPool2d((1, 1))

        self.C2 = Conv(32, 64)
        self.D2 = DownSampling(64)
        self.global_max_pool2 = nn.AdaptiveMaxPool2d((1, 1))

        self.C3 = Conv(64, 128)
        self.D3 = DownSampling(128)
        self.global_max_pool3 = nn.AdaptiveMaxPool2d((1, 1))


        self.C4 = Conv(128, 256)
        self.D4 = DownSampling(256)
        self.global_max_pool4 = nn.AdaptiveMaxPool2d((1, 1))

        self.C5 = Conv(256, 512)
        self.global_max_pool5 = nn.AdaptiveMaxPool2d((1, 1))

        # 4次上采样
        self.U1 = UpSampling(512)
        self.C6 = Conv(512, 256)
        self.global_max_pool6 = nn.AdaptiveMaxPool2d((1, 1))

        self.U2 = UpSampling(256)
        self.C7 = Conv(256, 128)
        self.global_max_pool7 = nn.AdaptiveMaxPool2d((1, 1))

        self.U3 = UpSampling(128)
        self.C8 = Conv(128, 64)
        self.global_max_pool8 = nn.AdaptiveMaxPool2d((1, 1))

        self.U4 = UpSampling(64)
        self.C9 = Conv(64, 32)
        self.global_max_pool9 = nn.AdaptiveMaxPool2d((1, 1))

        self.dropout = nn.Dropout(0.3)

        self.dense_layer = nn.Sequential(
            nn.Linear(1472, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 2)
        )

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(32, 1, 1, 1)

    def forward(self, x):
        # 下采样部分
        R1 = self.C1(x)
        global_features1 = self.global_max_pool1(R1).view(-1, 32)

        R2 = self.C2(self.D1(R1))
        global_features2 = self.global_max_pool2(R2).view(-1, 64)

        R3 = self.C3(self.D2(R2))
        global_features3 = self.global_max_pool3(R3).view(-1, 128)

        R4 = self.C4(self.D3(R3))
        global_features4 = self.global_max_pool4(R4).view(-1, 256)

        Y1 = self.C5(self.D4(R4))
        global_features5 = self.global_max_pool5(Y1).view(-1, 512)

        # 上采样部分
        # 上采样的时候需要拼接起来
        O1 = self.C6(self.U1(Y1, R4))
        global_features6 = self.global_max_pool6(O1).view(-1, 256)

        O2 = self.C7(self.U2(O1, R3))
        global_features7 = self.global_max_pool7(O2).view(-1, 128)

        O3 = self.C8(self.U3(O2, R2))
        global_features8 = self.global_max_pool8(O3).view(-1, 64)

        O4 = self.C9(self.U4(O3, R1))
        global_features9 = self.global_max_pool9(O4).view(-1, 32)

        # 拼接不同深度的特征
        concatenated_global_features = torch.cat(
            ( global_features1,global_features2, global_features3,
             global_features4,global_features5, global_features6,
             global_features7, global_features8,global_features9
             ), dim=1)

        concatenated_features = self.dropout(concatenated_global_features)

        # 使用密集层进行分类
        output = self.dense_layer(concatenated_features)

        return self.Th(self.pred(O4)), output

if __name__ == '__main__':
    net = UNet()


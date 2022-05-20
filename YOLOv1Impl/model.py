import torch.nn as nn

"""
模型网络架构的配置参数
如果是tuple，就表示(kernel_size, kernel_num, stride, padding) 
"""
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [
        (1, 256, 1, 0),
        (3, 512, 1, 1),
        4
    ],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [
        (1, 512, 1, 0),
        (3, 1024, 1, 1),
        2
    ],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNN_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class YOLOv1(nn.Module):
    def __init__(self, S, bbox_num):
        super(YOLOv1, self).__init__()
        self.net = self._create_network(architecture_config, S, bbox_num)
        print(self.net)

    def _create_network(self, net_config, S, bbox_num):
        layers = []
        # 开始的输入通道为3，RGB彩色图像
        in_channels = 3
        for l_cfg in net_config:
            # 类型是tuple，直接添加卷积层
            if type(l_cfg) == tuple:
                layers.append(
                    CNN_block(in_channels=in_channels, out_channels=l_cfg[1], kernel_size=l_cfg[0], stride=l_cfg[2],
                              padding=l_cfg[3])
                )
                # 记得更新in_channels，输入通道数是上一层的输出通道数
                in_channels = l_cfg[1]
            # 类型是str，添加最大池化层
            elif type(l_cfg) == str:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            # 类型是list，重复添加卷积层
            elif type(l_cfg) == list:
                # 获得重复次数
                repeat_times = l_cfg[-1]
                # 取出列表里的卷积层的配置信息
                l_cfg = l_cfg[:-1]
                for i in range(repeat_times):
                    for l in l_cfg:
                        layers.append(
                            CNN_block(in_channels=in_channels, out_channels=l[1], kernel_size=l[0], stride=l[2],
                                      padding=l[3])
                        )
                        # 记得更新输入通道
                        in_channels = l[1]

        # 最后添加全连接层
        layers.append(nn.Flatten(start_dim=1))
        layers.append(nn.Linear(in_features=S * S * 1024, out_features=4096))
        layers.append(nn.Linear(in_features=4096, out_features=S * S * (bbox_num * 5 + 20)))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

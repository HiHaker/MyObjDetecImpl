import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

"""
创建数据集类
应该重写 __len__ 以及 __getitem__ 方法
参数包括数据路径，S（划分的网格数）B（每个网格预测的bbox的数量）C（类别）
本项目使用的数据集的形式如下：
------------------------
-images
--xx.jpg
-labels
--xx.txt
-test.csv
-train.csv
------------------------
csv文件的形式为：
xx.jpg,xx.txt
"""


class VOCDataset(Dataset):
    def __init__(self, data_dir, S, B, C=20, train=True):
        super(VOCDataset, self).__init__()
        self.img_dir = data_dir+'/images'
        self.label_dir = data_dir+'/labels'
        self.S = S
        self.B = B
        self.C = C
        if train:
            self.annotations = pd.read_csv(data_dir + '/train.csv')
        else:
            self.annotations = pd.read_csv(data_dir + '/test.csv')

    def __len__(self):
        return len(self.annotations)

    """
    getitem方法要返回一张图像以及对应的标签，维度为[S,S,20+B*5]，具体就是[7,7,30]
    """
    def __getitem__(self, index):
        # 获得标签路径和图片路径
        label_path = self.label_dir + self.annotations.iloc[index, 1]
        img_path = self.img_dir + self.annotations.iloc[index, 0]

        # 处理标签数据
        # 打开标签文件
        boxes = []
        with open(label_path, 'r') as f:
            # readlines方法，一次读取所有内容后按行返回
            for line in f.readlines():
                # 去除字符串结尾的换行符
                line = line.replace("\n", "")
                params = line.split(" ")
                params = [int(x) if float(x) == int(float(x)) else float(x) for x in params]
                # 类别索引 中心点坐标 宽 高（均相对与整张图）
                boxes.append(params)

        # 打开图片文件
        img = Image.open(img_path)

        # 要对box的数据进行处理[cls, x, y, w, h]
        # 转化为tensor方便处理
        # boxes = torch.tensor(boxes)
        # 初始化要返回的label
        label = torch.zeros([self.S, self.S, self.B*5+20])
        # 把每一个box赋值到label张量里
        for box in boxes:
            # 计算出这个box在哪个grid cell
            row, col = int(box[2]*self.S), int(box[1]*self.S)
            # 计算box相对于当前cell的坐标
            x, y, w, h = self.S*box[1]-col, self.S*box[2]-row, self.S*box[3], self.S*box[4]
            # 赋值类别标签
            # 注意这里，因为每一个grid cell只负责预测一个框，所以这里label只赋值一个框的数据
            label[row][col][box[0]] = 1
            label[row][col][20:25] = 1, x, y, w, h

        return img, label


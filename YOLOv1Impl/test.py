from model import YOLOv1
import torch
import torch.nn as nn

net = YOLOv1(S=7, bbox_num=2)
imgs = torch.ones([2, 3, 448, 448])
result = net(imgs)
print(result.shape)


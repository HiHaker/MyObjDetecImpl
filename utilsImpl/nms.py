import torch

from iou import intersection_over_union

"""
NMS
参数：预测值preds 阈值threshold box_format
preds: [[cls_id, prob_score, x1, y1, x2, y2], ...]
"""


def non_max_suppression(preds, iou_threshold, prob_threshold, box_format="corners"):
    # 确保数据结构是列表
    assert type(preds) is list, "preds type is wrong!"

    # 首先对概率小于阈值的框抑制
    preds = [box for box in preds if box[1] > prob_threshold]
    # 匿名函数：以x[1]也就是概率值进行降序排序
    preds = sorted(preds, key=lambda x: x[1], reverse=True)
    preds_after_nms = []

    while preds:
        # 选出概率值最大的bbox
        chosen_box = preds.pop(0)
        # 进行比较，过滤
        # 这里box[0]表示的是类别id
        # 表示对检测同一个类别的框的集合进行过滤，对不同于当前框的类别的以及iou小于阈值的进行保留
        preds = [box for box in preds
                 if box[0] != chosen_box[0]
                 or intersection_over_union(
                torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), boxes_format=box_format
            )
                 < iou_threshold]

        preds_after_nms.append(chosen_box)

    return preds_after_nms

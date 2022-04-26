import torch

"""
计算IoU
参数有预测的bbox，真实的bbox，bbox的形式（是corner还是midpoint）
"""
def intersection_over_union(boxes_pred, boxes_label, boxes_format="midpoint"):
    # 根据box参数的不同形式来进行处理
    # midpoint
    if boxes_format == "midpoint":
        # 从中点变换到左上角corner：x坐标等于中点x减去1/2宽，y点坐标等于中点y减去1/2高
        box_pred_x1 = boxes_pred[..., 0] - boxes_pred[..., 2] / 2
        box_pred_y1 = boxes_pred[..., 1] - boxes_pred[..., 3] / 2
        box_label_x1 = boxes_label[..., 0] - boxes_label[..., 2] / 2
        box_label_y1 = boxes_label[..., 1] - boxes_label[..., 3] / 2
        # 从中点变换到右下角corner：x坐标等于中点x加上1/2宽，y点坐标等于中点y加上1/2高
        box_pred_x2 = boxes_pred[..., 0] + boxes_pred[..., 2] / 2
        box_pred_y2 = boxes_pred[..., 1] + boxes_pred[..., 3] / 2
        box_label_x2 = boxes_label[..., 0] + boxes_label[..., 2] / 2
        box_label_y2 = boxes_label[..., 1] + boxes_label[..., 3] / 2

    # corners
    else:
        box_pred_x1 = boxes_pred[..., 0]
        box_pred_y1 = boxes_pred[..., 1]
        box_label_x1 = boxes_label[..., 0]
        box_label_y1 = boxes_label[..., 1]
        box_pred_x2 = boxes_pred[..., 2]
        box_pred_y2 = boxes_pred[..., 3]
        box_label_x2 = boxes_label[..., 2]
        box_label_y2 = boxes_label[..., 3]

    # 计算union的坐标
    union_x1 = torch.max(box_pred_x1, box_label_x1)
    union_y1 = torch.max(box_pred_y1, box_label_y1)
    union_x2 = torch.min(box_pred_x2, box_label_x2)
    union_y2 = torch.min(box_pred_y2, box_label_y2)

    # 当预测框和真实框没有相交时可能会出现负数，所以限制值的范围最小为0
    intersection = (union_x2 - union_x1).clamp(0) * (union_y2 - union_y1).clamp(0)

    # 计算两个框的面积
    area_pred = (box_pred_x2 - box_pred_x1) * (box_pred_y2 - box_pred_y1)
    area_label = (box_label_x2 - box_label_x1) * (box_label_y2 - box_label_y1)

    return intersection / (area_pred + area_label - intersection + 1e-6)
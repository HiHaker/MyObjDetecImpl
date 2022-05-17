import torch
from collections import Counter
from iou import intersection_over_union

"""
计算mAP
参数：
pred_boxes：预测框
true_boxes：真实框GT
iou_threshold：iou阈值
box_format：box参数的格式，center或corner
num_class：类别数
"""
def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_class=20
):
    # step1
    # 得到预测结果

    # pred_boxes (list)：[[train_idx, class_pred, prob_score, x1, y1, x2, y2],...]
    average_precisions = []
    # 防止除零错误
    epsilon = 1e-6

    # 对于每一个物体类别，要计算它的AP
    for c in range(num_class):
        # 得到当前物体类别的预测框
        detections = [detection for detection in pred_boxes if detection[1] == c]
        # detections = []
        # for detection in pred_boxes:
        #     if detection[1] == c:
        #         detections.append(detection)

        # 得到当前物体类别的GT框
        ground_truths = [true_box for true_box in true_boxes if true_box[1] == c]
        # ground_truths = []
        # for true_box in true_boxes:
        #     if true_box[1] == c:
        #         ground_truths.append(true_box)

        # 统计当前这个类别每张图片的GT预测框的数量
        # img 0 has 3 bboxes
        # img 1 has 5 bboxes
        # amount_bboxes: {0: 3, 1: 5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # after doing it
        # amount_bboxes: {0: torch.tensor([0,0,0]), 1: torch.tensor([0,0,0,0,0])}

        # step2
        # 根据置信度进行降序排序
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        # 遍历每一个预测框
        # 并给每一个框赋值TP / FP
        for detection_idx, detection in enumerate(detections):
            # 找出当前预测框的图片对应的所有GT
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0

            # 计算当前预测框和每一个GT的IOU
            for idx, gt in enumerate(ground_truth_img):
                # 计算iou
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    boxes_format=box_format
                )

                # 更新最大的iou
                if iou > best_iou:
                    best_iou = iou
                    # 保存对应的GT的索引
                    best_gt_idx = idx

            # 如果最大的iou大于阈值
            if best_iou > iou_threshold:
                # 最大的iou对应的GT的值为0的话说明这个GT还没有给一个预测框赋予标签
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # 就可以赋予这个预测框TP的标签
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                # 否则，说明这个GT已经给一个预测框赋予了TP，那么只能给它赋予FP了
                # （对于一个GT,只能有一个TP的预测框）
                else:
                    FP[detection_idx] = 1
            # 如果小于阈值，也给它赋予FP的标签
            else:
                FP[detection_idx] = 1

        # [1, 1, 0, 1, 0, 1] -> [1, 2, 2, 3, 3, 4]
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        # 计算精确度和召回率
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        # 加上0，1这个点
        precisions = torch.cat(torch.tensor([1]), precisions)
        recalls = torch.cat(torch.tensor([0]), recalls)
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)
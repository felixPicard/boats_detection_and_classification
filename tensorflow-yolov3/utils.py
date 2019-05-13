import numpy as np
import pandas as pd

def real_boxes_size(boxes, ratio_w, ratio_h):
    for i in range(len(boxes)):
        boxes[i] = [boxes[i][0] * ratio_w, boxes[i][1] * ratio_h, boxes[i][2] * ratio_w, boxes[i][3] * ratio_h]

    return boxes


def IoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def score_values(GT_boxes, boxes):

    binary_labels = []
    FN, FP, TP = 0, 0, 0

    for gt_box in GT_boxes:

        if len(boxes) == 0:
            FN += 1
            continue

        iou = []
        for box in boxes:
            iou.append(IoU(box, gt_box))

        max_iou = np.argmax(np.array(iou))

        if iou[max_iou] > 0.5:
            TP += 1
            boxes = np.delete(boxes, max_iou, 0)
        else:
            FN += 1

    FP += len(boxes)
    return FN, FP, TP


def status(i, max):
    decades = float(i) / max * 10
    decades = int(decades)
    out = "[" + "=" * decades
    out += " " * (10 - decades) + "]"
    return out


def get_pandas_frame( threshold_list, precision_list, recall_list, f1_score_list):

    pandas_frame = np.array([['', 'Threshold', 'Precision', 'Recall', 'f1_score']])

    for j in range(len(precision_list)):
        row = [[j, threshold_list[j], precision_list[j], recall_list[j], f1_score_list[j]]]
        pandas_frame = np.append(pandas_frame, row, axis=0)

    return pd.DataFrame(data=pandas_frame[1:, 1:], index=pandas_frame[1:, 0], columns=pandas_frame[0, 1:])

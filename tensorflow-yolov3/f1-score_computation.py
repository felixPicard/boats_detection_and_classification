from scipy.io import loadmat
import numpy as np
import sys
import cv2
import time
import tensorflow as tf
from PIL import Image
from core import utils


# global variables
IMAGE_H, IMAGE_W = 416, 416
input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(),
                                                            "./checkpoint/yolov3_cpu_nms.pb",
                                                            ["Placeholder:0", "concat_9:0", "mul_6:0"])

classes = utils.read_coco_names('./data/coco.names')
num_classes = len(classes)

classification_table = ["Ferry", "Buoy", "Vessel/ship", "Speed boat", "Boat", "Kayak", "Sail boat", "Swimming person", "Flying bird/plane", "Other"]




# user variables
data = loadmat("../files/0799.mat")
video_path = "../files/videos/0799.avi"
threshold = 0.75
REAL_H, REAL_W = 1080, 1920


# functions global variables

ratio_h = float(REAL_H) / IMAGE_H
ratio_w = float(REAL_W) / IMAGE_W



def real_boxes_size(boxes):


    for i in range(len(boxes)):
        boxes[i] = [boxes[i][0]*ratio_w, boxes[i][1]*ratio_h, boxes[i][2]*ratio_w, boxes[i][3]*ratio_h]

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
            del boxes[max_iou]
        else:
            FN += 1

    FP = len(boxes)

    return FN, FP, TP


def f1_score(GT_boxes, boxes):

    FN, FP, TP = score_values(GT_boxes, boxes)

    if len(GT_boxes) == 0 and len(boxes) == 0:
        return 1
    if len(boxes) == 0 or len(GT_boxes) == 0:
        return 0

    precision = TP / float(TP + FP)
    recall = TP / float(TP + FN)

    if TP == 0:
        return 0

    return 2 * precision * recall / (precision + recall)

def yolo(current_frame):

    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(current_frame)

    img_resized = np.array(image.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
    img_resized = img_resized / 255.

    # sess is the current tensorflow session
    boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
    return  utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)



nb_of_frames = len(data['structXML'][0])
print("number of frames : ", nb_of_frames, ", meaning (30fps) : ", float(nb_of_frames)/30)

# frame_data[i] = xmlData of the i-th frame
frame_data = data['structXML'][0]

vid = cv2.VideoCapture(video_path)
i = 0

with tf.Session() as sess:

    f1 = 0

    while True:
        return_value, frame = vid.read()
        if not return_value:
            exit(-1)


        prediction_boxes, scores, labels = yolo(frame)

        if prediction_boxes is None:
            prediction_boxes = []

        for j in range(len(prediction_boxes)):
            if labels[j] != 8:
                del prediction_boxes[j]


        gt_boxes = []
        # computed boxes are not in the right shape
        original_size_boxes = real_boxes_size(prediction_boxes)

        if frame_data[i]['BB'].any():

            for j in range(len(frame_data[i]['BB'])):

                id = frame_data[i]['BB'][j] + 1
                x1 = int(id[0])
                y1 = int(id[1])
                x2 = int(id[0] + id[2])
                y2 = int(id[1] + id[3])


                gt_boxes.append([x1, y1, x2, y2])


            for gt_box in gt_boxes:
                cv2.rectangle(frame, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (255,255,255), 3)

            for prediction_box in prediction_boxes:
                cv2.rectangle(frame, (prediction_box[0], prediction_box[1]), (prediction_box[2], prediction_box[3]), (0, 255, 0), 3)

            f1 += f1_score(prediction_boxes, gt_boxes)

        cv2.namedWindow("jaaj")
        cv2.imshow("jaaj", frame)
        i+=1
        if cv2.waitKey(1) & 0xFF == ord('q'): break

        print(float(i)/ nb_of_frames * 100, "% done, f1_int = ", f1/i)

    print("f1 average score on ", i , " images : ", f1/i)

    cv2.destroyAllWindows()

vid.release()

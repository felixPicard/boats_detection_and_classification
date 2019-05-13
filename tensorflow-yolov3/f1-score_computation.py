from scipy.io import loadmat
import numpy as np
import sys
import cv2
import tensorflow as tf
from PIL import Image
from core import utils
from utils import *
import pandas as pd
from sklearn.metrics import precision_recall_curve

# global variables
IMAGE_H, IMAGE_W = 416, 416
input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(),
                                                            "./checkpoint/yolov3_cpu_nms.pb",
                                                            ["Placeholder:0", "concat_9:0", "mul_6:0"])

classes = utils.read_coco_names('./data/coco.names')
num_classes = len(classes)

classification_table = ["Ferry", "Buoy", "Vessel/ship", "Speed boat", "Boat", "Kayak", "Sail boat", "Swimming person",
                        "Flying bird/plane", "Other"]

# user variables
data_paths = ["../files/0799.mat", "../files/0797.mat", "../files/0801.mat"]
video_paths = ["../files/videos/0799.avi", "../files/videos/0797.avi", "../files/videos/0801.avi"]
REAL_H, REAL_W = 1080, 1920


# functions global variables
ratio_h = float(REAL_H) / IMAGE_H
ratio_w = float(REAL_W) / IMAGE_W


def yolo(current_frame):
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(current_frame)

    img_resized = np.array(image.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
    img_resized = img_resized / 255.

    # sess is the current tensorflow session
    boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
    return utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)


with tf.Session() as sess:

    for video_index in range(len(video_paths)):

        all_ground_truths = []
        all_predictions = []
        all_prediction_confidences = []

        precision, recall = [], []
        f1_scores = []

        i = 0
        data = loadmat(data_paths[video_index])
        video_path = video_paths[video_index]

        print("\nCurrent source : ", video_path)
        vid = cv2.VideoCapture(video_path)

        nb_of_frames = len(data['structXML'][0])
        print("number of frames : ", nb_of_frames, ", meaning (at 30fps) : ", float(nb_of_frames) / 30, "s")

        #  frame_data[i] = xmlData of the i-th frame
        frame_data = data['structXML'][0]

        f1_temp = 0
        stored_advance = 0
        sys.stdout.write('\r{} {}% done...'.format(status(0, 100), 0))

        while True:
            return_value, frame = vid.read()
            if not return_value:
                break

            prediction_boxes, scores, labels = yolo(frame)

            if prediction_boxes is None:
                prediction_boxes = []

            for j in range(len(prediction_boxes)):
                if labels[j] != 8:
                    del prediction_boxes[j]

            gt_boxes = []
            # computed boxes are not in the right shape
            original_size_boxes = real_boxes_size(prediction_boxes, ratio_w, ratio_h)

            #get the bounding boxes
            if frame_data[i]['BB'].any():

                for j in range(len(frame_data[i]['BB'])):
                    id = frame_data[i]['BB'][j] + 1
                    x1 = int(id[0])
                    y1 = int(id[1])
                    x2 = int(id[0] + id[2])
                    y2 = int(id[1] + id[3])

                    gt_boxes.append([x1, y1, x2, y2])

            for gt_box in gt_boxes:
                cv2.rectangle(frame, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (255, 255, 255), 3)


            for prediction_box in prediction_boxes:
               cv2.rectangle(frame, (prediction_box[0], prediction_box[1]), (prediction_box[2], prediction_box[3]),
                                  (0, 255, 0), 3)


            cv2.namedWindow("jaaj")
            cv2.imshow("jaaj", frame)
            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            # Plot of the loading bar
            advance = float(i) / nb_of_frames * 100
            if advance >= stored_advance + 1:
                sys.stdout.write('\r{} {}% done...'.format(status(stored_advance + 2, 100), stored_advance + 1))
                sys.stdout.flush()
                stored_advance = np.floor(advance)

            all_ground_truths.append(gt_boxes)
            all_predictions.append(prediction_boxes)
            all_prediction_confidences.append(scores)

        # flatten, sort, remove duplicates from all_confidences
        all_prediction_confidences_sorted = [prediction_confidence for prediction_confidence in all_prediction_confidences if prediction_confidence is not None]


        flattened_confidences = []
        for i in range(len(all_prediction_confidences_sorted)):
            for j in range(len(all_prediction_confidences_sorted[i])):
                flattened_confidences.append(all_prediction_confidences_sorted[i][j])

        all_prediction_confidences_sorted = flattened_confidences
        del flattened_confidences

        all_prediction_confidences_sorted.sort()
        all_prediction_confidences_sorted = list(dict.fromkeys(all_prediction_confidences_sorted))

        #all_prediction_confidences_sorted = np.linspace(0,1)

        # for each threshold
        stored_advance = 0


        print("\n\n Computing curves and scores")
        for threshold_index in range(len(all_prediction_confidences_sorted)-1):
            threshold = all_prediction_confidences_sorted[threshold_index]

            FN, FP, TP, = 0, 0, 0

            # Plot of the loading bar
            advance = float(threshold_index) / (len(all_prediction_confidences_sorted)-1) * 100
            if advance >= stored_advance + 1:
                sys.stdout.write('\r{} {}% done...'.format(status(stored_advance + 2, 100), stored_advance + 1))
                sys.stdout.flush()
                stored_advance = np.floor(advance)

            # for each frame
            for index in range(len(all_predictions)):
                considered_boxes = [all_predictions[index][j] for j in range(len(all_predictions[index])) if all_prediction_confidences[index][j] > threshold]
                if considered_boxes:
                    FN_temp, FP_temp, TP_temp = score_values(all_ground_truths[index], considered_boxes)
                    FN += FN_temp
                    FP += FP_temp
                    TP += TP_temp

            precision.append(TP / float(TP + FP))
            recall.append(TP / float(TP + FN))

            f1_scores.append(2*precision[-1]*recall[-1]/(precision[-1]+recall[-1]))

        current_pandas_frame = get_pandas_frame(all_prediction_confidences_sorted, precision, recall, f1_scores)
        current_pandas_frame.to_csv("csvFiles/pandas" + video_path[-8:-4] + ".csv", sep='\t')

    cv2.destroyAllWindows()


    vid.release()

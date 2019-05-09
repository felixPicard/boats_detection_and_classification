from scipy.io import loadmat
import numpy as np
import sys
import cv2


# user variables
data = loadmat("../files/0799.mat")
video_path = "../files/videos/0799.avi"
dataset_file = open("dataset.txt", "w")


classification_table = ["Ferry", "Buoy", "Vessel/ship", "Speed boat", "Boat", "Kayak", "Sail boat", "Swimming person", "Flying bird/plane", "Other"]

nb_of_frames = len(data['structXML'][0])
print("number of frames : ", nb_of_frames, ", meaning (30fps) : ", float(nb_of_frames)/30)

# motion_types[i] = xmlData of the i-th frame
motion_types = data['structXML'][0]

vid = cv2.VideoCapture(video_path)
i = 0

while True:
    return_value, frame = vid.read()
    if not return_value:
        exit(-1)

    if motion_types[i]['BB'].any():
        dataset_file.write(video_path[:-4] + "[" + str(i) + "].png ")

        for j in range(len(motion_types[i]['BB'])):

            id = motion_types[i]['BB'][j]+1
            x1 = int(id[0])
            y1 = int(id[1])
            x2 = int(id[0] + id[2])
            y2 = int(id[1] + id[3])

            class_index = classification_table.index(motion_types[i]['ObjectType'][j][0][0])

            dataset_file.write(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + str(class_index) + " ")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 3)

        dataset_file.write("\n")

    cv2.namedWindow("jaaj")
    cv2.imshow("jaaj", frame)
    i+=1
    if cv2.waitKey(1) & 0xFF == ord('q'): break


vid.release()
dataset_file.close()
cv2.destroyAllWindows()

#motions = [motion_types[i] for i in range(len(motion_types))]




"""
#motions = (obj for obj in motions if len(obj[0][0]) != 0)
for i in motion_types[0]:
    print(i)
    sys.stdout.flush()
#print([data['structXML'][0]['MotionType'][i][0][0][0] for i in range(len(data['structXML'][0]['MotionType']))])
"""
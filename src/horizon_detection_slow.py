import numpy as np
import cv2 as cv


def run():
    cap = cv.VideoCapture('../files/videos/0797.avi')

    while cap.isOpened():

        ret, image = cap.read()

        # todo fix the fact that the resolution of our image is divided by two
        # the imaged is re-sized at scale = 0.5, so that the computation is faster
        image = cv.resize(image, None, fx=0.5, fy=0.5)

        # color conversion
        imageTrans = cv.cvtColor(image, cv.COLOR_BGR2YCR_CB)

        # histogram normalization
        hist, bins = np.histogram(imageTrans.flatten(), 256, [0,256])
        cdf = hist.cumsum()

        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')

        # img2 is the image with the normalized histogram
        img2 = cdf[imageTrans]

        img2 = cv.cvtColor(img2, cv.COLOR_YCR_CB2BGR)
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        # helps filtering the noise
        img2 = cv.GaussianBlur(img2, (5,5), 0)

        # creates a border between the two parts
        _, img2 = cv.threshold(img2, 127, 255, cv.THRESH_BINARY)

        # filters noise
        kernel = np.ones((40,40), np.uint8)
        img2 = cv.erode(img2, kernel, 5)

        contours, _ = cv.findContours(img2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        cv.drawContours(image, contours, -1, (255,255,255), 3)
        image = cv.resize(image, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

        cv.namedWindow("Horizon detection")
        #frame = np.hstack((imgInt, img2))
        cv.imshow("Horizon detection", image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
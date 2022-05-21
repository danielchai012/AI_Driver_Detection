# import the necessary packages
from itertools import count
from tracemalloc import start
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import sys
import os
from PIL import Image, ImageDraw, ImageFont
import line_notify
import numpy as np


# construct the argument parser and parse the arguments
encoding_path = os.path.join(os.path.dirname(__file__), "encodings.pickle")
output_path = os.path.join(os.path.dirname(__file__), "output")

print(encoding_path)
# sys.path.append(os.getcwd())
# print(sys.path)


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def start_recon():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--encodings", required=False,
                    help="path to serialized db of facial encodings", default=encoding_path)
    ap.add_argument("-o", "--output", type=str,
                    help="path to output video", default=output_path)
    ap.add_argument("-y", "--display", type=int, default=1,
                    help="whether or not to display output frame to screen")
    ap.add_argument("-d", "--detection-method", type=str, default="hog",
                    help="face detection model to use: either `hog` or `cnn`")
    args = vars(ap.parse_args())
    # load the known faces and embeddings
    print("[INFO] loading encodings...")
    data = pickle.loads(open(args["encodings"], "rb").read())
    # initialize the video stream and pointer to output video file, then
    # allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    writer = None
    time.sleep(2.0)
    # loop over frames from the video file stream
    count = 0
    retry_count = 0
    retry_thresh = 3
    last_name = 'Not Same'
    namelist = []
    for name in data['names']:
        if name not in namelist:
            namelist.append(name)
    print(namelist)

    def hisEqulColor2(img):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe.apply(channels[0], channels[0])

        cv2.merge(channels, ycrcb)
        cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
        return img

    while True:
        # grab the frame from the threaded video stream
        frame = vs.read()

        # convert the input frame from BGR to RGB then resize it to have
        # a width of 750px (to speedup processing)
        fps_start = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)
        rgb = hisEqulColor2(rgb)
        r = frame.shape[1] / float(rgb.shape[1])
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        boxes = face_recognition.face_locations(rgb,
                                                model=args["detection_method"])
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding, tolerance=0.45)
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)
        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)
            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
        # if the video writer is None *AND* we are supposed to write
        # the output video to disk initialize the writer

        # check to see if we are supposed to display the output frame to
        # the screen
        fps_end = time.time()
        totalTime = fps_end - fps_start

        fps = 1 / totalTime
        # cv2.putText(frame, f'FPS: {int(fps)}', (20, 450),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        if len(names) > 0:
            i = set.intersection(set(names), set(namelist))
            if len(i) > 0:
                print('duplicates found')
                print(count)
                if count == 1:
                    last_name = names[0]
                if (count > 1) & (last_name != names[0]):
                    last_name = 'Not Same'
                    count = 0
            if (count > 1) & (last_name != names[0]):
                last_name = 'Not Same'
            print(names, count, last_name)
            count += 1

            if (count >= 10) & (last_name == names[0]):
                cv2.destroyAllWindows()
                vs.stop()
                return names
            elif (count >= 10) & (last_name == 'Not Same'):
                print('請重試')
                retry_count += 1
                count = 0

        else:
            count = 0
        if retry_count == retry_thresh:
            t = time.localtime()
            cv2.putText(frame, f'{time.strftime("%m/%d/%Y, %H:%M:%S", t)}', (0, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            p = os.path.sep.join([args["output"], "{}.png".format(
                str(time.strftime("%Y%m%d_%H%M%S", t)).zfill(1))])
            p = os.path.normpath(p)
            cv2.imwrite(p, frame)
            print(line_notify.lineNotifyMessage_Pic("非指定駕駛者", p))
            # print(line_notify.lineNotifyMessage("非指定駕駛者"))
            retry_count += 1
            print('已截圖上傳')
        if retry_count >= retry_thresh:
            cv2.putText(frame, f'{time.strftime("%m/%d/%Y, %H:%M:%S", t)}', (0, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame = cv2ImgAddText(frame, "已截圖上傳至LINE通報", 0, 0,
                                  textColor=(255, 0, 0), textSize=30)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # for i in range(names.count("Unknown")):
        #     if("Unknown" in names):
        #         names.remove("Unknown")
    # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            cv2.destroyAllWindows()
            vs.stop()
            return names

            # do a bit of cleanup
# check to see if the video writer point needs to be released
# if writer is not None:
#     writer.release()

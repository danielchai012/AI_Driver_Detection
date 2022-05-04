# coding=utf-8

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import threading
from threading import Timer
from threading import Thread
import numpy as np
import pyglet
import argparse
import imutils
import time
import dlib
import cv2
import os
import face_recognition
import mediapipe as mp


class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


def active_alarm():
    COUNTER += 1
    cv2.putText(frame, "COUNTER: {:.2f}".format(
        COUNTER), (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # 畫面列出閉眼時間
    cv2.putText(frame, "TIME: {:.2f}".format(
        TIME), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if (COUNTER == 1):
        start = time.time()
    if (COUNTER >= 1):
        end = time.time()
        TIME = (end - start)
        # print(TIME)
        # print("經過時間")

    # 如果閉眼時間大於設定閾值
    # 就觸發警報聲
    # if COUNTER >= EYE_AR_CONSEC_FRAMES:
    if TIME >= TIME_THRESH:
        # 如果警報沒開啟就打開

        if not ALARM_ON:
            ALARM_ON = True
            t = Thread(target=sound_alarm)
            t.daemon = True
            t.start()
            # timer = RepeatTimer(4, sound_alarm)
            # timer.start()
            # time.sleep(5)
            # timer.cancel()

        # 把警告文字顯示在畫面上
        cv2.putText(frame, "ALERT!!!!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 如果眼睛開合度沒低於設定閾值
    # 重置警告聲觸發和計時器
    else:
        COUNTER = 0
        ALARM_ON = False
        TIME = 0
        # timer.cancel()
        cv2.putText(frame, "TIME: {:.2f}".format(
            TIME), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def sound_alarm():
    # 播放警告聲

    pyglet.resource.path = ['C:\\Users\\CHAI\\Desktop\\eyesdetection']
    pyglet.resource.reindex()
    music = pyglet.resource.media('alarm.m4a')
    music.play()
    pyglet.app.run()
    print(threading.active_count())


def eye_aspect_ratio(eye):
    # 透過歐幾里得距離計算左右兩眼各垂直的距離
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # 透過歐幾里得距離計算左右兩眼各橫向的距離
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # 計算眼睛開合度
    ear = (A + B) / (2.0 * C)

    # 回傳EAR值
    return ear


ap = argparse.ArgumentParser()

ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())


EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 48
TIME_THRESH = 1

TIME = 0
COUNTER = 0
ALARM_ON = False
fullscreen = 0


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor 模型路徑
pwd = os.getcwd()  # 獲取當前路徑
alarm_path = os.path.join('C:/Users/CHAI/Desktop/eyesdetection')
model_path = os.path.join(
    'C:/Users/CHAI/Desktop/eyesdetection', 'model')  # 模型文件夾路徑
shape_detector_path = os.path.join(
    model_path, 'shape_predictor_68_face_landmarks.dat')  # 人臉特征點檢測模型路徑

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# 模型匯入
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_detector_path)

# 抓取左右兩眼的特征點的INDEX
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# 程式開始
print("[INFO] 開始程式·...")
vs = VideoStream(src=args["webcam"]).start()

time.sleep(1.0)

while True:
    # 抓取攝像機並重設圖片大小
    # 再轉成灰色
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = face_mesh.process(frame)
    fps_start = time.time()
    img_h, img_w, img_c = frame.shape
    face_3d = []
    face_2d = []
    # 抓取到的人臉轉為灰階
    rects = detector(gray, 0)
    # CNN
    # rects = face_recognition.api._raw_face_locations(
    #    gray, model='hog')
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See where the user's head tilting
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(
                nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(frame, p1, p2, (255, 0, 0), 3)

            # Add the text on the frame
            cv2.putText(frame, text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(frame, "x: " + str(np.round(x, 2)), (500, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "y: " + str(np.round(y, 2)), (500, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "z: " + str(np.round(z, 2)), (500, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
    # 讀取人臉
    for rect in rects:
        # 預測人臉特征點坐標位置, 再將其轉成Numpy 陣列
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 提取左右眼特征點的坐標，都進行眼睛開合度(EAR)計算
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # 平均兩眼EAR值
        ear = (leftEAR + rightEAR) / 2.0

        # 計算兩眼凸包再將其視覺化
        # 把眼睛線條畫出來
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # 檢查眼睛開合度有沒有低於設置閾值，有就開始計時
        if ear < EYE_AR_THRESH or text != "Forward":

            COUNTER += 1
            cv2.putText(frame, "COUNTER: {:.2f}".format(
                COUNTER), (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # 畫面列出閉眼時間
            cv2.putText(frame, "TIME: {:.2f}".format(
                TIME), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if (COUNTER == 1):
                start = time.time()
            if (COUNTER >= 1):
                end = time.time()
                TIME = (end - start)
                # print(TIME)
                # print("經過時間")

            # 如果閉眼時間大於設定閾值
            # 就觸發警報聲
            # if COUNTER >= EYE_AR_CONSEC_FRAMES:
            if TIME >= TIME_THRESH:
                # 如果警報沒開啟就打開

                if not ALARM_ON:
                    ALARM_ON = True
                    t = Thread(target=sound_alarm)
                    t.daemon = True
                    t.start()
                    # timer = RepeatTimer(4, sound_alarm)
                    # timer.start()
                    # time.sleep(5)
                    # timer.cancel()

                # 把警告文字顯示在畫面上
                cv2.putText(frame, "ALERT!!!!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 如果眼睛開合度沒低於設定閾值
        # 重置警告聲觸發和計時器
        else:
            COUNTER = 0
            ALARM_ON = False
            TIME = 0
            # timer.cancel()
            cv2.putText(frame, "TIME: {:.2f}".format(
                TIME), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 在畫面列出EAR值 眼睛開合度
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        fps_end = time.time()
        totalTime = fps_end - fps_start

        fps = 1 / totalTime
        #print("FPS: ", fps)

        cv2.putText(frame, f'FPS: {int(fps)}', (20, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    # show the frame
    out_win = "Frame"
    cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("f") and (fullscreen == 0):
        cv2.resizeWindow(out_win, 1024, 768)

        #cv2.setWindowProperty(out_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        fullscreen = 1

    elif key == ord("f") and (fullscreen == 1):
        cv2.resizeWindow(out_win, 640, 480)
        fullscreen = 0

    # 如果按q,離開迴圈並停止程式
    if key == ord("q"):
        break

# 清理程式
cv2.destroyAllWindows()
vs.stop()
os._exit(0)

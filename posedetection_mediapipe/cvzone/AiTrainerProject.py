"""import cv2
import numpy as np
import time
import PoseModule as pm

cap=cv2.VideoCapture("video.mp4")
detector=pm.PoseDetector()

while True:
    #img=cv2.imread("test.jpg")
    success,img=cap.read()
    img=cv2.resize(img,(1280,720))

    img=detector.findPose(img)
    lmlist=detector.findPosition(img,False)
    if len(lmlist)!=0:
        detector.findAngle(img,12,14,16)
        detector.findAngle(img,11,13,15)



    cv2.imshow("Image",img)
    cv2.waitKey(1)
    """
import cv2
import PoseModule as pm
# import mediapipe as mp
# import numpy as np
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
# # VIDEO FEED
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#     cv2.imshow('Mediapipe Feed', frame)
    
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
        
# cap.release()
# cv2.destroyAllWindows()
cap = cv2.VideoCapture(0)
detector = pm.PoseDetector()

while True:
    success, img = cap.read()
    if not success:
        break

    # img = cv2.resize(img, (1000, 600))
    img = detector.findPose(img)
    lmList, _ = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        angle1, img = detector.findAngle((lmList[12][0], lmList[12][1]),
                                         (lmList[14][0], lmList[14][1]),
                                         (lmList[16][0], lmList[16][1]),
                                         img=img)
        angle2, img = detector.findAngle((lmList[11][0], lmList[11][1]),
                                         (lmList[13][0], lmList[13][1]),
                                         (lmList[15][0], lmList[15][1]),
                                         img=img)

    cv2.imshow("Image", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

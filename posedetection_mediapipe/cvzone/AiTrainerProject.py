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
# cap = cv2.VideoCapture(0)
# detector = pm.PoseDetector()

# while True:
#     success, img = cap.read()
#     if not success:
#         break

#     # img = cv2.resize(img, (1000, 600))
#     img = detector.findPose(img)
#     lmList, _ = detector.findPosition(img, draw=False)
    
#     if len(lmList) != 0:
#         angle1, img = detector.findAngle((lmList[12][0], lmList[12][1]),
#                                          (lmList[14][0], lmList[14][1]),
#                                          (lmList[16][0], lmList[16][1]),
#                                          img=img)
#         angle2, img = detector.findAngle((lmList[11][0], lmList[11][1]),
#                                          (lmList[13][0], lmList[13][1]),
#                                          (lmList[15][0], lmList[15][1]),
#                                          img=img)

#     if len(lmList) != 0:
#         # Calculate angle for right arm (shoulder, elbow, wrist)
#         angle1, img = detector.findAngle((lmList[12][0], lmList[12][1]),
#                                           (lmList[14][0], lmList[14][1]),
#                                           (lmList[16][0], lmList[16][1]),
#                                           img=img)        # Calculate angle for left arm (shoulder, elbow, wrist)

#         angle2, img = detector.findAngle((lmList[11][0], lmList[11][1]),
#                                           (lmList[13][0], lmList[13][1]),
#                                           (lmList[15][0], lmList[15][1]),
#                                           img=img)
#         # Display angles on the image
#         cv2.putText(img, str(int(angle1)), 
#                     (lmList[14][1] - 50, lmList[14][2] + 50), 
#                     cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
#         cv2.putText(img, str(int(angle2)), 
#                     (lmList[13][1] - 50, lmList[13][2] + 50), 
#                     cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

#         # Evaluate form based on angles
#         feedback = ""
#         if 190 < angle1 < 350 and 190 < angle2 < 350:
#             feedback = "Good form"
#         else:
#             feedback = "Correct your form"

#         # Display feedback on the image
#         cv2.putText(img, feedback, (50, 50), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

#     # Display the image
#     cv2.imshow("Image", img)
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# cap.release()
import cv2
import PoseModule as pm

cap = cv2.VideoCapture(0)
detector = pm.PoseDetector()

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (1280, 720))

    # Find pose and landmarks
    img = detector.findPose(img)
    lmList, _ = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        # Check if user's body is within a certain range of the center of the screen
        center_range = 100  # Adjust this range as needed

        # Calculate the center of the screen
        img_height, img_width, _ = img.shape
        center_x = img_width // 2

        # Check if neck landmark is within the range around the center
        is_within_center_range = (lmList[1][0] >= center_x - center_range) and (lmList[1][0] <= center_x + center_range)

        # Check if whole body is visible
        neck_to_hip_dist = abs(lmList[8][1] - lmList[1][1])
        whole_body_visible = (neck_to_hip_dist > 0) and (lmList[1][1] > 0 and lmList[1][1] < img_height) and (lmList[8][1] > 0 and lmList[8][1] < img_height)

        if is_within_center_range and whole_body_visible:
            # Calculate angles for arms
            angle1, img = detector.findAngle((lmList[12][0], lmList[12][1]),
                                             (lmList[14][0], lmList[14][1]),
                                             (lmList[16][0], lmList[16][1]),
                                             img=img)

            angle2, img = detector.findAngle((lmList[11][0], lmList[11][1]),
                                             (lmList[13][0], lmList[13][1]),
                                             (lmList[15][0], lmList[15][1]),
                                             img=img)

            # Display angles on the image
            cv2.putText(img, str(int(angle1)), 
                        (lmList[14][1] - 50, lmList[14][2] + 50), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(img, str(int(angle2)), 
                        (lmList[13][1] - 50, lmList[13][2] + 50), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            # Evaluate form based on angles
            feedback = ""
            if 190 < angle1 < 350 and 190 < angle2 < 350:
                feedback = "Good form"
            else:
                feedback = "Correct your form"

            # Display feedback on the image
            cv2.putText(img, feedback, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            # Display message if user is not within the center range or whole body is not visible
            cv2.putText(img, "Please stay within the center range and ensure full body visibility", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

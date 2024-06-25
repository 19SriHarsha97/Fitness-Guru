import cv2
import PoseModule as pm

# Initialize video capture
cap = cv2.VideoCapture("cvzone/video.mp4")
detector = pm.PoseDetector()

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (1000, 700))  # Resize for better visibility
    img = detector.findPose(img)  # Find the pose
    lmList, _ = detector.findPosition(img, draw=False)  # Find landmarks without drawing

    if len(lmList) != 0:
        # Calculate angle for right arm (shoulder, elbow, wrist)
        angle1, img = detector.findAngle((lmList[12][0], lmList[12][1]),
                                          (lmList[14][0], lmList[14][1]),
                                          (lmList[16][0], lmList[16][1]),
                                          img=img)        # Calculate angle for left arm (shoulder, elbow, wrist)

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
        if 182 < angle1 < 350 and 182 < angle2 < 350:
            feedback = "Good form"
        else:
            feedback = "Correct your form"

        # Display feedback on the image
        cv2.putText(img, feedback, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Display the image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

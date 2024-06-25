import cv2
import PoseModule as pm
import threading

class PoseEstimator:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = pm.PoseDetector()
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def capture_frame(self):
        while self.running:
            success, img = self.cap.read()
            if success:
                with self.lock:
                    self.frame = cv2.resize(img, (640, 480))

    def check_form(self, lmList):
        feedback = []

        # Define form criteria for hammer curls
        good_form_min_angle = 190  # Lower bound for good form during curl
        good_form_max_angle = 320  # Upper bound for good form during curl
        
        # Calculate angles for arms
        angle_right_elbow, _ = self.detector.findAngle((lmList[12][0], lmList[12][1]),
                                                       (lmList[14][0], lmList[14][1]),
                                                       (lmList[16][0], lmList[16][1]))

        angle_left_elbow, _ = self.detector.findAngle((lmList[11][0], lmList[11][1]),
                                                      (lmList[13][0], lmList[13][1]),
                                                      (lmList[15][0], lmList[15][1]))

        # Check if both elbows are within the good form angle range
        if not (good_form_min_angle <= angle_right_elbow <= good_form_max_angle):
            feedback.append("Right arm angle out of optimal range")

        if not (good_form_min_angle <= angle_left_elbow <= good_form_max_angle):
            feedback.append("Left arm angle out of optimal range")

        # Check if the back is straight
        # Landmarks: 11 (left shoulder), 12 (right shoulder), 23 (left hip), 24 (right hip)
        shoulder_midpoint = ((lmList[11][0] + lmList[12][0]) // 2, (lmList[11][1] + lmList[12][1]) // 2)
        hip_midpoint = ((lmList[23][0] + lmList[24][0]) // 2, (lmList[23][1] + lmList[24][1]) // 2)

        # Calculate the angle of the torso (straight line between shoulders and hips)
        torso_angle = self.calculate_angle(shoulder_midpoint, hip_midpoint)

        # Define the range for a straight back (you might need to adjust these values)
        min_torso_angle = 80  # Close to vertical
        max_torso_angle = 100  # Close to vertical

        if not (min_torso_angle <= torso_angle <= max_torso_angle):
            feedback.append("Keep your back straight")

        return feedback, angle_right_elbow, angle_left_elbow, torso_angle

    def calculate_angle(self, point1, point2):
        # Calculate the angle between two points
        x1, y1 = point1
        x2, y2 = point2
        angle = abs(cv2.fastAtan2(y2 - y1, x2 - x1))
        return angle

    def process_frame(self):
        while self.running:
            if self.frame is None:
                continue

            with self.lock:
                img = self.frame.copy()

            img = self.detector.findPose(img)
            lmList, _ = self.detector.findPosition(img, draw=False)
            
            if len(lmList) != 0:
                center_range = 100
                img_height, img_width, _ = img.shape
                center_x = img_width // 2

                is_within_center_range = (lmList[1][0] >= center_x - center_range) and (lmList[1][0] <= center_x + center_range)
                neck_to_hip_dist = abs(lmList[8][1] - lmList[1][1])
                whole_body_visible = (neck_to_hip_dist > 0) and (lmList[1][1] > 0 and lmList[1][1] < img_height) and (lmList[8][1] > 0 and lmList[8][1] < img_height)

                if is_within_center_range and whole_body_visible:
                    feedback, angle_right_elbow, angle_left_elbow, torso_angle = self.check_form(lmList)

                    # Display the angles on the image
                    cv2.putText(img, f"Right Elbow: {int(angle_right_elbow)}", 
                                (lmList[14][0] - 50, lmList[14][1] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(img, f"Left Elbow: {int(angle_left_elbow)}", 
                                (lmList[13][0] - 50, lmList[13][1] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(img, f"Torso: {int(torso_angle)}", 
                                (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # Display feedback on the image
                    for i, msg in enumerate(feedback):
                        cv2.putText(img, msg, (50, 150 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    if not feedback:
                        cv2.putText(img, "Good form", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                else:
                    cv2.putText(img, "Please stay within the center range and ensure full body visibility", 
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

    def run(self):
        thread1 = threading.Thread(target=self.capture_frame)
        thread2 = threading.Thread(target=self.process_frame)
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        self.cap.release()
        cv2.destroyAllWindows()

pose_estimator = PoseEstimator()
pose_estimator.run()

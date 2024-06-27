import cv2
import mediapipe as mp
import threading
import numpy as np
import json

# Step 2: Function to Read the Dataset from a JSON File
def load_workout_dataset(file_path):
    with open(file_path, 'r') as file:
        workout_dataset = json.load(file)
    return workout_dataset

# Load the dataset
workout_dataset = load_workout_dataset('posedetection_mediapipe/cvzone/workout_dataset.json')

# Generalized function to access the dataset
def get_workout_info(workout_name):
    if workout_name in workout_dataset:
        return workout_dataset[workout_name]
    else:
        raise ValueError("Workout not found in dataset")

class PoseEstimator3D:
    def __init__(self, workout_name):
        self.cap = cv2.VideoCapture(0)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.prev_wrist_positions = []
        self.workout_info = get_workout_info(workout_name)

    def capture_frame(self):
        while self.running:
            success, img = self.cap.read()
            if success:
                with self.lock:
                    self.frame = cv2.resize(img, (640, 480))

    def check_form(self, landmarks):
        feedback = []
        
        angle_ranges = self.workout_info['angle_range']
        relevant_landmarks = self.workout_info['landmarks']
        
        # Calculate angles for arms
        angle_right_elbow = self.calculate_angle(landmarks[relevant_landmarks['right_elbow'][0]], landmarks[relevant_landmarks['right_elbow'][1]], landmarks[relevant_landmarks['right_elbow'][2]])
        angle_left_elbow = self.calculate_angle(landmarks[relevant_landmarks['left_elbow'][0]], landmarks[relevant_landmarks['left_elbow'][1]], landmarks[relevant_landmarks['left_elbow'][2]])

        # Check if both elbows are within the good form angle range
        if not (angle_ranges['right_elbow'][0] <= angle_right_elbow <= angle_ranges['right_elbow'][1]):
            feedback.append("Right arm angle out of optimal range")

        if not (angle_ranges['left_elbow'][0] <= angle_left_elbow <= angle_ranges['left_elbow'][1]):
            feedback.append("Left arm angle out of optimal range")
        
        # Check if back is straight
        back_angle_tolerance = 30  # Allowable deviation from 180 degrees
        angle_back = self.calculate_angle(landmarks[11], landmarks[23], landmarks[25])
        if angle_back < (180 - back_angle_tolerance) or angle_back > (180 + back_angle_tolerance):
            feedback.append("Keep your back straight")
        
        # Check if person is not swinging arms
        swing_threshold = 0.1  # Adjust this threshold as needed
        if len(self.prev_wrist_positions) >= 5:
            right_wrist_disp = self.calculate_displacement(self.prev_wrist_positions, 'right')
            left_wrist_disp = self.calculate_displacement(self.prev_wrist_positions, 'left')
            if right_wrist_disp > swing_threshold:
                feedback.append("Avoid swinging your right arm")
            if left_wrist_disp > swing_threshold:
                feedback.append("Avoid swinging your left arm")
            self.prev_wrist_positions.pop(0)
        
        self.prev_wrist_positions.append({
            'right': (landmarks[15].x, landmarks[15].y),
            'left': (landmarks[16].x, landmarks[16].y)
        })

        return feedback, angle_right_elbow, angle_left_elbow

    def calculate_angle(self, a, b, c):
        a = [a.x, a.y, a.z]
        b = [b.x, b.y, b.z]
        c = [c.x, c.y, c.z]

        ba = [a[i] - b[i] for i in range(3)]
        bc = [c[i] - b[i] for i in range(3)]

        cosine_angle = sum(ba[i] * bc[i] for i in range(3)) / ((sum(ba[i] ** 2 for i in range(3)) ** 0.5) * (sum(bc[i] ** 2 for i in range(3)) ** 0.5))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)
    
    def calculate_displacement(self, positions, wrist):
        displacements = [np.sqrt((positions[i+1][wrist][0] - positions[i][wrist][0])**2 + (positions[i+1][wrist][1] - positions[i][wrist][1])**2) for i in range(len(positions) - 1)]
        return np.mean(displacements)

    def process_frame(self):
        while self.running:
            if self.frame is None:
                continue

            with self.lock:
                img = self.frame.copy()

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self.pose.process(img_rgb)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark

                # Draw landmarks
                self.mp_drawing.draw_landmarks(img, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

                feedback, angle_right_elbow, angle_left_elbow = self.check_form(landmarks)

                # Display the angles on the image
                cv2.putText(img, f"Right Elbow: {int(angle_right_elbow)}", 
                            (int(landmarks[13].x * img.shape[1]), int(landmarks[13].y * img.shape[0] - 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(img, f"Left Elbow: {int(angle_left_elbow)}", 
                            (int(landmarks[14].x * img.shape[1]), int(landmarks[14].y * img.shape[0] - 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Display feedback on the image
                for i, msg in enumerate(feedback):
                    cv2.putText(img, msg, (50, 150 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                if not feedback:
                    cv2.putText(img, "Good form", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

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

pose_estimator = PoseEstimator3D("hammer_curls")
pose_estimator.run()

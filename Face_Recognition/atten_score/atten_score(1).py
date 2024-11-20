import cv2
import face_recognition
import dlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from imutils import face_utils

# Initialize dlib's face detector and the facial landmark predictor
p = r"C:\Users\Praneeth Kumar\Pictures\Prani\VS Code(New)\Face_Recognition(old)\facial-landmarks-recognition\shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(p)

# Create a directory to save screenshots
if not os.path.exists("Praneeth_screenshots_Atten_score"):
    os.makedirs("Praneeth_screenshots_Atten_score")

# Load the known image for recognition
known_image = face_recognition.load_image_file("4.jpg")
known_faces = face_recognition.face_encodings(known_image, num_jitters=50, model='large')[0]

# Create a DataFrame to store recognized face information
columns = ['Name', 'Date', 'Time', 'Screenshot', 'Attentive', 'Attention Score']
df = pd.DataFrame(columns=columns)

# Set up camera capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define the codec and create VideoWriter object to save the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('praneeth_atten_score.mp4', fourcc, 20.0, (frame_width, frame_height))

# Define thresholds for yaw and pitch
MAX_YAW_THRESHOLD = 0.5
MAX_PITCH_THRESHOLD = 0.5

def get_head_pose(landmarks):
    # Define points for head pose estimation
    image_points = np.array([landmarks[30], landmarks[8], landmarks[36], landmarks[45], landmarks[48], landmarks[54]], dtype="double")
    model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-165.0, 170.0, -135.0), (165.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)])
    camera_matrix = np.array([[320, 0, 160], [0, 320, 120], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    return rotation_vector, translation_vector

def calculate_attention_score(yaw, pitch):
    yaw_score = max(0, 1 - abs(yaw[0]) / MAX_YAW_THRESHOLD)
    pitch_score = max(0, 1 - abs(pitch[0]) / MAX_PITCH_THRESHOLD)
    return (yaw_score + pitch_score) / 2

frame_count = 0
last_save_time = datetime.now()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame")
            break

        frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distance = face_recognition.face_distance([known_faces], face_encoding)[0]

            if distance < 0.6:
                name = 'Praneeth Kumar'
                now = datetime.now()
                
                face_landmarks = landmark_predictor(gray, dlib.rectangle(left, top, right, bottom))
                landmarks = [(p.x, p.y) for p in face_landmarks.parts()]
                
                # Head pose calculation
                rotation_vector, translation_vector = get_head_pose(landmarks)
                yaw, pitch, roll = rotation_vector
                attention_score = calculate_attention_score(yaw, pitch)
                attentive = 'Yes' if attention_score >= 0.5 else 'No'
                
                # Annotate frame and take screenshot if attentive
                screenshot_filename = f"Praneeth_screenshots_Atten_score/{name}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                cv2.putText(frame, f'Attentive (Score: {attention_score:.2f})' if attentive == 'Yes' else f'Not Attentive (Score: {attention_score:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if attentive == 'Yes' else (0, 0, 255), 2, cv2.LINE_AA)

                if attentive == 'Yes':
                    cv2.imwrite(screenshot_filename, frame)
                
                new_entry = pd.DataFrame({
                    'Name': [name],
                    'Date': [now.strftime("%Y-%m-%d")],
                    'Time': [now.strftime("%H:%M:%S")],
                    'Screenshot': [screenshot_filename],
                    'Attentive': [attentive],
                    'Attention Score': [attention_score]
                })
                df = pd.concat([df, new_entry], ignore_index=True)

                # Draw rectangle around face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0) if attentive == 'Yes' else (0, 0, 255), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        # Write the frame to the video file
        out.write(frame)
        
        # Display the frame
        cv2.imshow("Video Stream", frame)

        # Save the DataFrame to Excel every 10 seconds
        if (datetime.now() - last_save_time).total_seconds() >= 5:
            df.to_excel('praneeth_attendance_with_attention_score.xlsx', index=False)
            print("DataFrame saved to 'praneeth_attendance_with_attention_score.xlsx'")
            last_save_time = datetime.now()

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    if not df.empty:
        df.to_excel('praneeth_attendance_with_attention_score.xlsx', index=False)
        print("Final attendance with attentiveness saved to 'praneeth_attendance_with_attention_score.xlsx'.")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

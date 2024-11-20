import cv2 as cv
import face_recognition

# Load the known image of Guru Teja
known_image = face_recognition.load_image_file("4.jpg")
known_faces = face_recognition.face_encodings(known_image, num_jitters=50, model='large')[0]

# Launch the live camera
cam = cv.VideoCapture(0)

# Check if camera is opened
if not cam.isOpened():
    print("Camera not working")
    exit()

# Set up the video writer with the desired filename
frame_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
output_filename = "praneeth_face_recog.mp4"
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

# Confidence threshold
confidence_threshold = 0.6  # Adjust this value as needed

# When camera is opened
while True:
    # Capture the image frame-by-frame
    ret, frame = cam.read()
    
    # Check if frame is reading or not
    if not ret:
        print("Can't receive the frame")
        break

    # Face detection in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame)

    recognized = False

    for face_encoding in face_encodings:
        # Compute the distance to the known face encoding
        distance = face_recognition.face_distance([known_faces], face_encoding)[0]

        if distance < confidence_threshold:  # Check if the distance is below the threshold
            recognized = True
            # Get the location of the face
            for face_location in face_locations:
                top, right, bottom, left = face_location
                cv.rectangle(frame, (left, top), (right, bottom), color=(0, 255, 0), thickness=2)
                frame = cv.putText(frame, 'Praneeth Kumar', (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                                   (255, 0, 0), 2, cv.LINE_AA)

    if not recognized:
        frame = cv.putText(frame, 'Not Praneeth Kumar', (30, 55), cv.FONT_HERSHEY_SIMPLEX, 1,
                           (255, 0, 0), 2, cv.LINE_AA)

    # Write the frame to the output video
    out.write(frame)

    # Display the resulting frame
    cv.imshow('Video Stream', frame)

    # End the streaming
    if cv.waitKey(1) == ord('q'):
        break

# Release the capture and the video writer
cam.release()
out.release()
cv.destroyAllWindows()

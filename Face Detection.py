import cv2
from deepface import DeepFace

# Load Haarcascade for face detection
face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open the camera
cam = cv2.VideoCapture(0)

while True:
    ret, video_data = cam.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        face = video_data[y:y+h, x:x+w]
        cv2.rectangle(video_data, (x, y), (x+w, y+h), (0, 255, 0), 2)

        try:
            # Analyze the detected face for emotions
            analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            # Extract dominant emotion
            emotion = analysis[0]['dominant_emotion']
            # Display the detected emotion
            cv2.putText(video_data, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"Emotion Detection Error: {e}")

    # Show the video feed with face and emotion
    cv2.imshow("Face and Emotion Detection", video_data)

    # Break the loop on pressing 's'
    if cv2.waitKey(10) == ord("s"):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()

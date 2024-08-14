import cv2
from deepface import DeepFace
import csv
from datetime import datetime

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def analyze_emotions():
    cap = cv2.VideoCapture(0)

    # Open CSV file for writing results in real-time
    with open('emotions_log.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'dominant_emotion', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))

            # Convert frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Extract the face ROI (Region of Interest)
                face_roi = frame[y:y + h, x:x + w]

                try:
                    # Analyze emotions using DeepFace on the face ROI
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    if isinstance(result, list):
                        result = result[0]  # If the result is a list, take the first element
                    
                    # Get the current timestamp
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Write the results to the CSV file
                    writer.writerow([
                        timestamp,
                        result['dominant_emotion'],
                        result['emotion']['angry'],
                        result['emotion']['disgust'],
                        result['emotion']['fear'],
                        result['emotion']['happy'],
                        result['emotion']['sad'],
                        result['emotion']['surprise'],
                        result['emotion']['neutral']
                    ])

                    # Draw rectangle around face and label with predicted emotion
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, result['dominant_emotion'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                except Exception as e:
                    print(f"Error: {e}")

            # Display the resulting frame
            cv2.imshow('Real-time Emotion Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    analyze_emotions()

import cv2
from fer import FER
import csv
from datetime import datetime
import tensorflow as tf

# Enable GPU acceleration if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU will be used for analysis.")
    except RuntimeError as e:
        print(f"Error: {e}")
else:
    print("No GPU found. Using CPU.")

# Initialize the FER emotion detector
emotion_detector = FER(mtcnn=True)

# Parameters for optimizing processing
FRAME_RESIZE = (480, 360)  # Resize frame for faster processing
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for considering an emotion
SMOOTHING_FRAMES = 5  # Number of frames to average over for smoothing

def analyze_emotions():
    cap = cv2.VideoCapture(0)

    # Write results in real-time in CSV file
    with open('emotions_log.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'dominant_emotion', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])

        # Frame buffer for smoothing
        emotion_queue = []

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Resize frame for faster processing
            frame = cv2.resize(frame, FRAME_RESIZE)

            # Detect emotions using the FER detector
            result = emotion_detector.detect_emotions(frame)
            
            if result:
                for face in result:
                    (x, y, w, h) = face['box']
                    emotions = face['emotions']
                    dominant_emotion = max(emotions, key=emotions.get)

                    # Apply confidence threshold
                    if emotions[dominant_emotion] >= CONFIDENCE_THRESHOLD:
                        emotion_queue.append(dominant_emotion)
                        
                        if len(emotion_queue) > SMOOTHING_FRAMES:
                            emotion_queue.pop(0)

                        # Smoothing the detected emotion over the last few frames
                        smoothed_emotion = max(set(emotion_queue), key=emotion_queue.count)

                        # Get the current timestamp
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # Write the results to the CSV file
                        writer.writerow([
                            timestamp,
                            smoothed_emotion,
                            emotions['angry'],
                            emotions['disgust'],
                            emotions['fear'],
                            emotions['happy'],
                            emotions['sad'],
                            emotions['surprise'],
                            emotions['neutral']
                        ])

                        # Draw rectangle around face and label with predicted emotion
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, smoothed_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Display the resulting frame
            cv2.imshow('Real-time Emotion Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    analyze_emotions()

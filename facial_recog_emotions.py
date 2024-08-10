import cv2
from deepface import DeepFace
import pandas as pd
from datetime import datetime

def analyze_emotions():
    # Start video capture
    cap = cv2.VideoCapture(0)

    # A list to store the results before converting to Dataframe
    data = []

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Frame analysis using DeepFace
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list):
                result = result[0]  # If the result is a list, take the first element
            
            # Get the current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Add the results to the data list
            data.append({
                "timestamp": timestamp,
                "dominant_emotion": result['dominant_emotion'],
                "angry": result['emotion']['angry'],
                "disgust": result['emotion']['disgust'],
                "fear": result['emotion']['fear'],
                "happy": result['emotion']['happy'],
                "sad": result['emotion']['sad'],
                "surprise": result['emotion']['surprise'],
                "neutral": result['emotion']['neutral']
            })

            # Display the resulting frame
            cv2.putText(frame, 
                        result['dominant_emotion'], 
                        (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2, 
                        cv2.LINE_AA)
        except Exception as e:
            print(f"Error: {e}")

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

    # Convert list to a DataFrame
    df = pd.DataFrame(data)

    # Save as a CSV file
    df.to_csv("emotions_log.csv", index=False)

if __name__ == "__main__":
    analyze_emotions()

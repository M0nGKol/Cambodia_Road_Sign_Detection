"""
Real-time Traffic Sign Detection using Webcam
Author: Your Name
Description: Test trained YOLO model with webcam feed
"""

import cv2
from ultralytics import YOLO
import time

# Configuration
MODEL_PATH = 'runs/detect/cambodia_traffic_signs/weights/best.pt'  # Update with your model path
CONFIDENCE_THRESHOLD = 0.5
WEBCAM_ID = 0  # 0 for default webcam, 1 for external

# Class names based on your dataset
CLASS_NAMES = {
    0: 'Prohibitory',
    1: 'Mandatory',
    2: 'Priority',
    3: 'Warning',
    4: 'Service',
    5: 'Other'
}

def main():
    """Main function to run webcam detection"""
    
    # Load the trained model
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(WEBCAM_ID)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set webcam resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Starting webcam... Press 'q' to quit")
    
    # FPS calculation variables
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Run inference
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Get the annotated frame
        annotated_frame = results[0].plot()
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter >= 10:
            fps_end_time = time.time()
            fps = fps_counter / (fps_end_time - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        
        # Display FPS on frame
        cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display detection count
        num_detections = len(results[0].boxes)
        cv2.putText(annotated_frame, f'Detections: {num_detections}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Traffic Sign Detection', annotated_frame)
        
        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f'screenshot_{timestamp}.jpg'
            cv2.imwrite(filename, annotated_frame)
            print(f"Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam test completed")

if __name__ == "__main__":
    main()
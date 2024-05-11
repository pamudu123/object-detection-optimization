'''
This script performs object detection on a series of videos.

The user can interact with the video playback using keyboard commands:
- Press 'd' to move to the next video in the sequence.
- Press 'a' to move to the previous video in the sequence.
- Press 'q' to quit the video playback.

'''

import cv2
import numpy as np
import time
from utils import get_video_paths
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')

video_directory = 'alarms'
video_paths = get_video_paths(video_directory)
print(video_paths)
current_index = 0
video_count = len(video_paths)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    exit_flag = False  # Flag to control the outer loop
    cap = cv2.VideoCapture(video_paths[current_index])
    t_new = 0
    t_prev = 0
    fps_sum = 0.0  # Initialize the sum of FPS values
    frame_count = 0  # Initialize the frame count

    print("=" * 25)
    print(f'{current_index}/{video_count} : {video_paths[current_index]}')

    while True:
        ret, frame = cap.read()
        if not ret:
            # Move to the next video once one video is over
            current_index = (current_index + 1) % video_count
            break
        
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=0.25, iou=0.7, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8], half=True, device=0, verbose = False)
        annotated_frame = results[0].plot()

        # calculate fps
        t_new = time.time()
        fps = 1 / (t_new - t_prev)
        t_prev = t_new

        # Display FPS on the top-left corner
        annotated_frame[:30, :150] = (0, 0, 0)
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(annotated_frame, fps_text, (10, 20), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Update the FPS sum and frame count
        fps_sum += fps
        frame_count += 1

        cv2.imshow('Video Analyzer', annotated_frame)
        key = cv2.waitKey(5)
        if key == ord('d'):
            # Move to the next video
            current_index = (current_index + 1) % video_count
            break
        elif key == ord('a'):
            # Move to the previous video
            current_index = (current_index - 1) % video_count
            break
        elif key == ord('q'):
            # Quit the video player
            exit_flag = True  # Set the flag to True to exit the outer loop
            break

    if exit_flag:
        # Check if the flag is set to True
        break

    # Calculate and print the average FPS for the current video
    if frame_count > 0:
        avg_fps = fps_sum / frame_count
        print(f"Average FPS : {avg_fps:.2f}")

    # Release the video capture object
    cap.release()

cv2.destroyAllWindows()
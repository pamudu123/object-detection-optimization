"""
This script is designed to profile the performance of the YOLO object detection model on a series of videos. 
It measures and records several performance metrics including frame processing speed (FPS), GPU usage, 
and the number of detected objects per frame. The results are saved in a pickle file for later analysis.

Outputs:
- A pickle file ('video_stats_yolov8m.pkl') that contains a list of dictionaries. Each dictionary corresponds 
  to a video and contains the following keys:
  - 'path': The file path of the video.
  - 'fps': A list of FPS measurements for each frame.
  - 'gpu_usage': A list of GPU usage measurements for each frame.
  - 'objects_per_frame': A list of the count of detected objects for each frame.
  - 'resolution': The resolution of the video (width, height).
  - 'duration_sec': The duration of the video in seconds.
  - 'duration_frames': The total number of frames in the video.
  - 'channels': The number of channels in the video frames (typically 3 for RGB).
  - 'avg_fps': The average FPS over all frames in the video.
  - 'avg_gpu_usage': The average GPU usage over all frames in the video.
  - 'processing_time': The total time taken to process the video.
- Annotated videos saved in the local dir
"""

import cv2
import time
import os
import pynvml
import pickle
from ultralytics import YOLO
from memory_profiler import profile
from utils import get_video_paths

# Initialize GPU monitoring
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Object detection classes
classes = [
    '0: person', '1: bicycle', '2: car', '3: motorcycle',
    '4: airplane', '5: bus', '6: train', '7: truck', '8: boat'
]
list_of_classes = list(range(9))

@profile
def process_video(video_path, model, output_video_path):
    cap = cv2.VideoCapture(video_path)
    t_new = 0
    t_prev = 0
    fps_sum = 0.0
    frame_count = 0
    gpu_usage_sum = 0.0
    font = cv2.FONT_HERSHEY_SIMPLEX


    # To record video stats
    video_stats = {
        'path': video_path,
        'fps': [],
        'gpu_usage': [],
        'objects_per_frame': []
    }
    video_stats['resolution'] = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # calc duration of the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # Get total number of frames
    fps = cap.get(cv2.CAP_PROP_FPS)      # Get frames per second (FPS)
    duration_seconds = total_frames / fps 
    video_stats['duration_sec'] = duration_seconds
    video_stats['duration_frames'] = total_frames
    n_channels = None

    # For save result video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if not n_channels:
            n_channels = frame.shape[2]
            video_stats['channels'] = n_channels

        # Run YOLOv8 inference on the frame
        model_output = model(frame, conf=0.25, iou=0.7, classes=list_of_classes, half=True, device=0, verbose=False)

        annotated_frame = model_output[0].plot()

        # Calculate FPS
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

        # Get GPU usage
        gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
        gpu_usage_sum += gpu_usage

        video_stats['fps'].append(fps)
        video_stats['gpu_usage'].append(gpu_usage)
        video_stats['objects_per_frame'].append(len(model_output[0].boxes.data))

        out.write(annotated_frame)

        # comment out for profiling
        cv2.imshow('Video Analyzer', annotated_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Calculate and return the average FPS and GPU usage for the current video
    if frame_count > 0:
        video_stats['avg_fps'] = fps_sum / frame_count
        video_stats['avg_gpu_usage'] = gpu_usage_sum / frame_count
    
    return video_stats

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')

# Get video paths
video_directory = 'alarms'
video_paths = get_video_paths(video_directory)

output_dir = r'annotated_videos'
os.makedirs(output_dir, exist_ok=True)

# Process each video and collect profiling data
results = []
for i, video_path in enumerate(video_paths):
    print(f'{i+1}/{len(video_paths)} | {video_path}')
    output_video_path = os.path.join(output_dir, f"{os.path.split(video_path)[1].split('.')[0]}.mp4")

    t_start = time.perf_counter()
    video_stats = process_video(video_path, model, output_video_path)
    t_stop = time.perf_counter()

    video_stats['processing_time'] = t_stop - t_start
    # print(video_stats)
    results.append(video_stats)

# Shut down GPU monitoring
pynvml.nvmlShutdown()

# Save video_stats for analysis
with open('video_stats_yolov8m.pkl', 'wb') as f:
    pickle.dump(results, f)




"""
This script optimizes video processing for object detection using these techniques:

1. Multithreading:
   - Two threads are used to enhance processing efficiency:
     * One thread is dedicated to extracting frames from the video.
     * Another thread runs the object detection model.
   - These threads communicate via a queue, which helps in maintaining a consistent flow of frames for detection.

2. Model Selection:
   - The YOLOv8 nano model is employed instead of the larger YOLOv8 medium model.
   - This change prioritizes higher frame rates (fps) and reduced GPU resource usage over maximal accuracy.

3. TensorRT Conversion:
   - The model is optimized using NVIDIA's TensorRT, enhancing inference speed on NVIDIA GPUs.
   - This optimization involves layer fusion, precision calibration, and kernel auto-tuning to boost performance.

4. Uniform Frame Size:
   - All video frames are resized to 640x480 pixels.
   - This uniformity allows the GPU to process batches of frames more efficiently, leveraging the parallel nature of GPU computations.

5. Precision Reduction:
   - FP16 (half-precision) model weights are used to decrease memory demands and speed up computation on GPUs.
   - For scenarios where CPU optimization is needed, UINT8 precision is preferred to minimize computational overhead.

6. Frame Skipping:
   - To reduce the computational load, every alternate frame is skipped.
   - This approach assumes that consecutive frames are similar enough that skipping does not significantly affect the overall detection quality.

These optimizations collectively enhance the processing speed, making the system suitable for real-time applications where high throughput and efficiency are crucial.
"""

import cv2
import time
import threading
from queue import Queue
from utils import get_video_paths, log_results


from ultralytics import YOLO
from memory_profiler import profile
import pynvml

# Initialize GPU monitoring
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def video_capture_thread(video_path, queue):
    cap = cv2.VideoCapture(video_path)

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process only every second frame
        if frame_id % 2 == 0:
            # Resize the frame
            resized_frame = cv2.resize(frame, (640, 480))
            # Add the frame to the queue
            queue.put(resized_frame)
        
        frame_id += 1

    # Put a sentinel to indicate end of video
    queue.put(None)
    cap.release()


def detection_thread(queue, model):
    fps_sum = 0.0
    frame_count = 0
    gpu_usage_sum = 0.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    t_prev = time.time()

    # Object detection classes
    # '0: person', '1: bicycle', '2: car', '3: motorcycle',
    # '4: airplane', '5: bus', '6: train', '7: truck', '8: boat'
    list_of_classes = list(range(9))

    while True:
        frame = queue.get()
        if frame is None:
            break
        
        # Run YOLOv8 inference on the frame
        model_out = model(frame, conf=0.25, iou=0.7, classes=list_of_classes, half=True, device=0, verbose=False)
        annotated_frame = model_out[0].plot()

        # Calculate FPS
        t_new = time.time()
        fps = 2 / (t_new - t_prev)   # skip 1 frame for each processing frame
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

        cv2.imshow('Video Analyzer', annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    
    # Print average statistics
    if frame_count > 0:
        avg_fps = fps_sum / frame_count
        avg_gpu_usage = gpu_usage_sum / frame_count

        print(f"Average FPS: {avg_fps:.4f}")
        print(f"Average GPU Usage: {avg_gpu_usage:.4f}%")

        log_results(f"Average FPS: {avg_fps:.4f} | Average GPU Usage: {avg_gpu_usage:.4f}%")


# Load the YOLOv8 model
model = YOLO('yolov8n.engine')

# Frame queue
frame_queue = Queue()

# Get video paths
video_directory = 'alarms'
video_paths = get_video_paths(video_directory)


for i, video_path in enumerate(video_paths):
    print(f"{i}/{len(video_paths)} | {video_path}")
    log_results(f"== {i}/{len(video_paths)} | {video_path} ==")

    # Start the video capture thread
    capture_thread = threading.Thread(target=video_capture_thread, args=(video_path, frame_queue))
    capture_thread.start()

    # Start the detection thread
    detect_thread = threading.Thread(target=detection_thread, args=(frame_queue, model))
    detect_thread.start()

    # Wait for both threads to finish
    capture_thread.join()
    detect_thread.join()

# Shut down GPU monitoring
pynvml.nvmlShutdown()
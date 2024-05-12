# YOLOv8 Runtime Optimization

This covers various optimization techniques applied to the YOLOv8 object detector to enhance its performance when processing video streams. The optimizations are designed to improve frames per second (fps), reduce GPU usage, and maintain effective object detection.

## Scripts Overview

- **run_yolov8.py**: Runs the YOLO object detector on a series of videos, allowing navigation back and forth through the video frames.
  
- **yolov8_profiling.py**: Profiles the YOLOv8 object detection to gather performance metrics.

- **results_analysis.py**: Analyzes the data collected during code profiling to provide insights into performance improvements.

- **yolo_optimization.py**: Implements various optimization techniques to enhance the performance of the YOLOv8 object detection.

## Performance Metrics

For memory profiling, the `memory_profiler` Python library is used. Key metrics include:

1. **avg_fps**: Average frames per second, indicating how smoothly video frames are being processed.

2. **avg_gpu_usage**: Average GPU usage for processing a video.

3. **avg_objects_per_frame**: Average number of objects detected per frame, providing insights into the scene's complexity and the effectiveness of object detection algorithms.

Additionally, the NVIDIA GPU Profiler (version 1.07a3_x64) is used for resource monitoring.

## Optimization Techniques

1. **Multithreading for Efficiency**:
   - **Thread 1**: Extracts frames from the video.
   - **Thread 2**: Runs the object detection model on the frames.
   - These threads communicate via a queue, balancing the load between frame extraction and processing.

2. **Model Selection for Speed**:
   - Switched from YOLOv8 medium model to YOLOv8 nano for higher fps and fewer GPU resources, at a slight cost to accuracy.

3. **TensorRT Conversion**:
   - The model is converted using TensorRT, an SDK for high-performance deep learning inference, optimizing the model to run faster on NVIDIA GPUs.

4. **Uniform and Fixed Frame Size**:
   - All frames are resized to 640x480 pixels, the native image size for YOLOv8, to avoid dynamic memory allocation during inference.

5. **Precision Reduction**:
   - Model weights are converted to FP16 (half-precision floating-point) to reduce memory usage and computational demands.
   - For CPU optimization, UINT8 precision might be used.

6. **Frame Skipping**:
   - Every alternate frame is skipped, assuming consecutive frames are similar, reducing workload without significantly affecting analysis quality.

These optimizations lead to a nearly threefold increase in fps, allowing quicker processing of video streams.

## Note on Video Processing

- **Video Resolution**:
  - High-resolution increases processing time due to more data per frame.
  - The YOLO algorithm processes high-resolution frames slower than lower-resolution ones.

- **Number of Objects Detected**:
  - The YOLO algorithm divides the image into a grid, predicting bounding boxes and class probabilities for each grid cell in one pass.
  - More objects might require more post-processing time to filter out low-confidence detections or overlapping bounding boxes.

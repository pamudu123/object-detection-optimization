# Surveillance Setup Overview

This outlines the design considerations for a surveillance setup utilizing four video feeds. The setup is optimized for real-time processing with a focus on balancing speed and accuracy.

## Model Selection and Optimization

### YOLOv8 Nano with TensorRT and FP16 Quantization

- **Why YOLOv8 Nano?**  
  YOLOv8 Nano is selected for its balance between accuracy and speed. As the smallest variant in the YOLOv8 series, it offers faster inference times due to fewer parameters, albeit at a slight cost to accuracy.

- **TensorRT and FP16 Quantization:**  
  Using a TensorRT-optimized model with FP16 quantization accelerates inference by leveraging the GPU more efficiently. This helps to maintain high processing speeds without significant loss of accuracy.

## Video Preprocessing

### Fixed Frame Size

- **Consistent Input Dimensions:**  
  All video frames are resized to 640x480 before being processed. This ensures that the model receives consistently sized input, which is crucial for optimal GPU performance. The choice of 640x480 aligns with the native image size for YOLOv8.


## Parallel Processing Setup

### Multi-Threading

- **Separate Model per Thread:**  
  Four separate YOLOv8 Nano models are initialized for the four video feeds. This design increases GPU memory usage but also enhances the overall frames per second (fps). The increased memory usage is within acceptable limits, making this a viable design choice.

- **Benefits of Parallel Processing:**  
  Using multi-threading allows each thread to handle a different segment of the video/sources concurrently, significantly reducing total processing time compared to a single-threaded approach.

## Resource Monitoring

### GPU Monitoring with PyNVML

- **Resource Tracking:**  
  The system uses PyNVML to monitor GPU utilization, logging usage statistics for diagnostics and performance tuning.


## Video Output and Display

### Stacked Video Output

- **Unified View:**  
  The processed frames from all feeds are stacked into a single window for display. This combined frame is also saved into a single output video file, offering a comprehensive view of all video feeds in real time.


## Error Handling and System Robustness

### Graceful Exit and Error Handling

- **Robust Processing Loop:**  
  The pipeline includes mechanisms to handle cases where video frames are not available (e.g., end of the video), allowing the system to exit gracefully. This robustness is crucial for real-world applications.
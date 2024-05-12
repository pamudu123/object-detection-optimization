"""
Video Processing System using YOLOv8 and OpenCV

This script is designed to process multiple video feeds concurrently using the YOLOv8 model for object detection
and OpenCV for video handling. It segments video paths into four sets, processes them in separate threads,
and visualizes the results in a unified window. The script also measures and displays the performance statistics
like FPS and GPU usage.

Pipeline:
    - Initialize YOLO models and queues for each video source.
    - Divide the video paths into four sets and assign them to four sources.
    - Start processing threads for each source.
    - Start a GUI update thread to visualize the results.
    - Wait for all processing to complete and join all threads.

Output:
    - A window displaying the processed video feeds in a 2x2 grid.
    - FPS and GPU usage overlay on each video feed.
    - A saved pickle file containing detailed processing statistics.
    - The script handles user interruption and safely closes all resources.
"""

from ultralytics import YOLO
import cv2
import numpy as np
import threading
import queue
import time
import pickle
import pynvml
from memory_profiler import profile
from utils import get_video_paths
from stack_Images import stackImages


# Get video paths
video_directory = 'alarms'
video_paths = get_video_paths(video_directory)

# Calculate the length of each portion
# sample_size = 5
sample_size = int(len(video_paths) // 4)
video_set_1 = video_paths[:sample_size]
video_set_2 = video_paths[sample_size:sample_size*2]
video_set_3 = video_paths[sample_size*2:sample_size*3]
video_set_4 = video_paths[sample_size*3:sample_size*4]

# Sources
source1 = video_set_1
source2 = video_set_2
source3 = video_set_3
source4 = video_set_4


# setup model
yolo_model_path = r'yolov8n.engine'
model1 = YOLO(yolo_model_path, task='detect')
model2 = YOLO(yolo_model_path, task='detect')
model3 = YOLO(yolo_model_path, task='detect')
model4 = YOLO(yolo_model_path, task='detect')


# Queue to communicate between threads
queue1 = queue.Queue()
queue2 = queue.Queue()
queue3 = queue.Queue()
queue4 = queue.Queue()

SOURCE_IDS = [1, 2, 3, 4]


def start_new_video(video_path):
    """Initialize video capture and calculate video stats."""
    cap = cv2.VideoCapture(video_path)
    video_stats = {
        'video_path': video_path,
        'resolution': (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        'duration_sec': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
        'duration_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    return cap, video_stats


@profile
def run_in_thread(video_paths, model, id, output_queue):
    """Thread function to process each video and perform object detection."""

    # Initialize GPU monitoring
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    # Object detection classes
    #'0: person', '1: bicycle', '2: car', '3: motorcycle',
    # '4: airplane', '5: bus', '6: train', '7: truck', '8: boat'
    list_of_classes = list(range(9))

    print(f"----------- {id} starts -----------")
    video_id = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    t_prev = time.time()

    cap, video_stats = start_new_video(video_paths[video_id])

    while True:
        # Read the video frames
        ret, frame = cap.read()

        if not ret:
            # After one video start next one
            video_id += 1
            if video_id >= len(video_paths):
                video_stats["live"] = False 
                output_queue.put((id, None, video_stats))
                break
            else:
                video_path = video_paths[video_id]
                print(f'{video_id}/{len(video_paths)} || {video_path}')
                cap, video_stats = start_new_video(video_path)
                ret, frame = cap.read()


        video_stats["live"] = True  # TO check the video is completed or not
        # Resize the frame to 640x480
        frame = cv2.resize(frame, (640, 480))

        # Run YOLOv8 inference
        results = model(frame, conf=0.25, iou=0.7, classes=list_of_classes, half=True, device=0, verbose=False)
        annotated_frame = results[0].plot()

        # Calculate FPS
        t_new = time.time()
        fps = 1 / (t_new - t_prev)
        t_prev = t_new

        # Display FPS on the top-left corner
        annotated_frame[:30, :150] = (0, 0, 0)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 20), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Get GPU usage
        gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu

        video_stats['fps'] = fps
        video_stats['gpu_usage'] = gpu_usage
        video_stats['objects_per_frame'] = len(results[0].boxes.data)

        # Put the frame in the queue
        output_queue.put((id, annotated_frame, video_stats))

    cap.release()


def get_process_output():
    """Main loop to process and display the output from all threads."""
    ## Initialize variables for detections
    display_frame1 = None
    display_frame2 = None   
    display_frame3 = None
    display_frame4 = None

    results = {SOURCE_IDS[0]:
                {'video_path' : [],
                 'resolution' : [],
                 'duration_sec' : [],
                 'duration_frames' : [],
                 'fps': [],
                 'gpu_usage': [],
                 'objects_per_frame': []
                },
            SOURCE_IDS[1]:
                {'video_path' : [],
                 'resolution' : [],
                 'duration_sec' : [],
                 'duration_frames' : [],
                 'fps': [],
                'gpu_usage': [],
                'objects_per_frame': []
                },
            SOURCE_IDS[2]:
                {'video_path' : [],
                 'resolution' : [],
                 'duration_sec' : [],
                 'duration_frames' : [],
                 'fps': [],
                'gpu_usage': [],
                'objects_per_frame': []
                },
            SOURCE_IDS[3]:
                {'video_path' : [],
                 'resolution' : [],
                 'duration_sec' : [],
                 'duration_frames' : [],
                 'fps': [],
                'gpu_usage': [],
                'objects_per_frame': []
                },
            }

    live1 = True
    live2 = True   
    live3 = True
    live4 = True

    # Get frame dimensions
    display_factor = 1
    frame_width = int(640 * display_factor * 2)
    frame_height = int(480 * display_factor* 2)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
    out = cv2.VideoWriter('result_video2.mp4', fourcc, 60, (frame_width, frame_height))
   
    while True:
        try:
            id1, display_frame1, video1_stats = queue1.get_nowait()
            if video1_stats['live']:
                results[id1]['video_path'].append(video1_stats['video_path'])
                results[id1]['resolution'].append(video1_stats['resolution'])
                results[id1]['duration_sec'].append(video1_stats['duration_sec'])
                results[id1]['duration_frames'].append(video1_stats['duration_frames'])
                results[id1]['fps'].append(video1_stats['fps'])
                results[id1]['gpu_usage'].append(video1_stats['gpu_usage'])
                results[id1]['objects_per_frame'].append(video1_stats['objects_per_frame'])
            else:
                live1 = False

        except queue.Empty:
            pass

        try:
            id2, display_frame2, video2_stats = queue2.get_nowait()
            if video2_stats['live']:
                results[id2]['video_path'].append(video2_stats['video_path'])
                results[id2]['resolution'].append(video2_stats['resolution'])
                results[id2]['duration_sec'].append(video2_stats['duration_sec'])
                results[id2]['duration_frames'].append(video2_stats['duration_frames'])
                results[id2]['fps'].append(video2_stats['fps'])
                results[id2]['gpu_usage'].append(video2_stats['gpu_usage'])
                results[id2]['objects_per_frame'].append(video2_stats['objects_per_frame'])
            else:
                live2 = False

        except queue.Empty:
            pass

        try:
            id3, display_frame3, video3_stats = queue3.get_nowait()
            if video3_stats['live']:
                results[id3]['video_path'].append(video3_stats['video_path'])
                results[id3]['resolution'].append(video3_stats['resolution'])
                results[id3]['duration_sec'].append(video3_stats['duration_sec'])
                results[id3]['duration_frames'].append(video3_stats['duration_frames'])
                results[id3]['fps'].append(video3_stats['fps'])
                results[id3]['gpu_usage'].append(video3_stats['gpu_usage'])
                results[id3]['objects_per_frame'].append(video3_stats['objects_per_frame'])
            else:
                live3 = False
        except queue.Empty:
            pass

        try:
            id4, display_frame4, video4_stats = queue4.get_nowait()
            if video4_stats['live']:
                results[id4]['video_path'].append(video4_stats['video_path'])
                results[id4]['resolution'].append(video4_stats['resolution'])
                results[id4]['duration_sec'].append(video4_stats['duration_sec'])
                results[id4]['duration_frames'].append(video4_stats['duration_frames'])
                results[id4]['fps'].append(video4_stats['fps'])
                results[id4]['gpu_usage'].append(video4_stats['gpu_usage'])
                results[id4]['objects_per_frame'].append(video4_stats['objects_per_frame'])
            else:
                live4 = False
        except queue.Empty:
            pass

        ### dsiplay screens
        display_frames = [display_frame1, display_frame2, display_frame3, display_frame4]
        
        display_rows = []
        
        for i in range(len(display_frames)):
            if display_frames[i] is None:
                display_frames[i] = np.zeros((480, 640, 3), dtype=np.uint8)

        display_rows = [[display_frames[0], display_frames[1]],
                        [display_frames[2], display_frames[3]]]

        # save entire 4 frames as one
        img_stack = stackImages(display_factor, display_rows)
        out.write(img_stack)

        cv2.imshow("Video Feed", img_stack)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
        if sum([live1, live2, live3, live4]) == 0:
            print("====FINISHED====")

            # Save video_stats for analysis
            with open('video_stats.pkl', 'wb') as f:
                pickle.dump(results, f)

            break
    
    out.release()
    cv2.destroyAllWindows()



# Initialize threads
source_thread1 = threading.Thread(target=run_in_thread, args=(source1, model1, SOURCE_IDS[0], queue1), daemon=True)
source_thread2 = threading.Thread(target=run_in_thread, args=(source2, model2, SOURCE_IDS[1], queue2), daemon=True)
source_thread3 = threading.Thread(target=run_in_thread, args=(source3, model3, SOURCE_IDS[2], queue3), daemon=True)
source_thread4 = threading.Thread(target=run_in_thread, args=(source4, model4, SOURCE_IDS[3], queue4), daemon=True)


# Start threads
source_thread1.start()
source_thread2.start()
source_thread3.start()
source_thread4.start()


# Start the GUI update thread
process_thread = threading.Thread(target=get_process_output, daemon=True)
process_thread.start()

# Wait for threads to finish
source_thread1.join()
source_thread2.join()
source_thread3.join()
source_thread4.join()

process_thread.join()
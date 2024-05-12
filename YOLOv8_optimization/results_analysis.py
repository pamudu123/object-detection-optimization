import pickle
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean
import math

# Load the results data from the pickle file
results_file = r'YOLOv8_optimization\results\video_stats_yolov8m_e4.pkl'
with open(results_file, 'rb') as f:
   results = pickle.load(f)

# Sort the results based on average FPS in descending order
results = sorted(results, key=lambda x: x['avg_fps'], reverse=True)

# Initialize an empty dictionary to store video data
video_data = {}

# Get all the unique keys from the results
all_keys = set().union(*results)

# Initialize lists for each key in the video_data dictionary
for key in all_keys:
   video_data[key] = []

# Populate the video_data dictionary with the data from results
for video_stats in results:
   for key in all_keys:
       video_data[key].append(video_stats.get(key, None))

# Calculate the average objects per frame (rounded up to the nearest integer)
avg_objects_per_frame = [math.ceil(mean(obj_count)) for obj_count in video_data['objects_per_frame']]

# Initialize empty lists to store video data
video_paths = []
video_resolutions = []
avg_fps_values = []
process_times = []
gpu_mean_loads = []
tot_frames = []
channels_list = []
video_durations_sec = []

# Populate the lists with the corresponding data from results
for video_stats in results:
   video_path = video_stats['path']
   video_resolution = video_stats['resolution']
   avg_fps = video_stats['avg_fps']
   avg_gpu_usage = video_stats['avg_gpu_usage']
   processing_time = video_stats['processing_time']
   video_duration_sec = video_stats['duration_sec']
   tot_frame_count = video_stats['duration_frames']
   channels = video_stats['channels']

   video_paths.append(video_path)
   video_resolutions.append(video_resolution)
   avg_fps_values.append(avg_fps)
   process_times.append(processing_time)
   gpu_mean_loads.append(avg_gpu_usage)
   tot_frames.append(tot_frame_count)
   video_durations_sec.append(video_duration_sec)
   channels_list.append(channels)

# Calculate the processing time per frame
process_times_per_frame = [process_time / duration for process_time, duration in zip(process_times, tot_frames)]

# Create a pandas DataFrame with the processed data
df_data = {
   'Video Path': video_paths,
   'Resolution': video_resolutions,
   'Channels': channels_list,
   'Total Frames': tot_frames,
   'Average FPS': avg_fps_values,
   'Average Objects per Frame': avg_objects_per_frame,
   'Processing Time per Frame': process_times_per_frame
}

df = pd.DataFrame(df_data)

# Save the DataFrame to an Excel file
excel_file_path = 'results_1.xlsx'
df.to_excel(excel_file_path, index=False)

# Print the worst-case videos (lowest FPS)
print("Worst case videos (lowest FPS):")
for i in range(len(avg_fps_values)):
   print(f"Video Path: {video_paths[i]}, Average FPS: {round(avg_fps_values[i], 4)}, Average Objects per Frame: {avg_objects_per_frame[i]}, Processing Time per Frame: {round(process_times_per_frame[i], 4)}")

# Print the best-case videos (highest FPS)
print("\nBest case videos (highest FPS):")
for i in range(len(avg_fps_values)-1, max(len(avg_fps_values)-40, 0), -1):
   print(f"Average FPS: {avg_fps_values[i]}, Average Objects per Frame: {avg_objects_per_frame[i]}, Video Path: {video_paths[i]}")

# Scatter plot: Average Objects per Frame vs. FPS
plt.figure(figsize=(8, 6))
plt.scatter(avg_objects_per_frame, avg_fps_values, label='Average Objects per Frame', marker='o', color='blue')
plt.xlabel('Average Objects per Frame')
plt.ylabel('FPS')
plt.title('FPS vs. Average Objects per Frame')
plt.legend()
plt.tight_layout()
plt.show()

# Scatter plot: Processing Time per Frame vs. FPS
plt.figure(figsize=(8, 6))
plt.scatter(process_times_per_frame, avg_fps_values, label='Processing Time per Frame', marker='s', color='red')
plt.xlabel('Processing Time per Frame')
plt.ylabel('FPS')
plt.title('FPS vs. Processing Time per Frame')
plt.legend()
plt.tight_layout()
plt.show()

# Print average case videos
print("\nAverage case videos:")
avg_case_idx = len(video_paths) // 2
for i in range(avg_case_idx-5, avg_case_idx+5):
   if i >= len(video_paths):
       break
   print(f"Video Path: {video_paths[i]}, Average FPS: {round(avg_fps_values[i], 4)}, Average Objects per Frame: {avg_objects_per_frame[i]}, Processing Time per Frame: {round(process_times_per_frame[i], 4)}")

# Scatter plot: Video Durations (seconds) vs. Processing Times
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].scatter(video_durations_sec, process_times)
axs[0].set_xlabel('Video Durations (seconds)')
axs[0].set_ylabel('Processing Times')
axs[0].set_title('Processing Times vs. Video Durations (Seconds)')

# Scatter plot: Video Durations (frames) vs. Processing Times
axs[1].scatter(tot_frames, process_times)
axs[1].set_xlabel('Video Durations (frames)')
axs[1].set_ylabel('Processing Times')
axs[1].set_title('Processing Times vs. Video Durations (Frames)')
plt.tight_layout()
plt.show()

# Histogram: Distribution of GPU Mean Loads
plt.figure(figsize=(8, 6))
plt.hist(gpu_mean_loads, bins=10, color='skyblue', edgecolor='black')
plt.xlabel('GPU Mean Load (%)')
plt.ylabel('Frequency')
plt.title('Distribution of GPU Mean Loads')
plt.grid(True)
plt.show()

# Scatter plot: GPU Load vs. Duration (frames)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(tot_frames, gpu_mean_loads, color='skyblue')
plt.xlabel('Duration (Frames)')
plt.ylabel('GPU Load (%)')
plt.title('GPU Load vs. Duration')

# Scatter plot: GPU Load vs. Processing Time per Frame
plt.subplot(1, 2, 2)
plt.scatter(process_times_per_frame, gpu_mean_loads, color='salmon')
plt.xlabel('Processing Time (seconds)')
plt.ylabel('GPU Load (%)')
plt.title('GPU Load vs. Processing Time Per Frame')
plt.tight_layout()
plt.show()
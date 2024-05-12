import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results from the pickle file
with open(r'C:\Users\PK\Desktop\projects\promisQ\video_stats_4videos_tt.pkl', 'rb') as f:
    results = pickle.load(f)

# Extract feeds
feed1 = results[1]
feed2 = results[2]
feed3 = results[3]
feed4 = results[4]

# Function to separate data by video path
def separate_video_wise(data, feed_id):
    separated_data = {}
    
    # Iterate over each frame index
    for i in range(len(data['video_path'])):
        video_path = data['video_path'][i]
        
        # Initialize data structure if video_path is not a key
        if video_path not in separated_data:
            separated_data[video_path] = {
                'feed_id': feed_id,
                'resolution': data['resolution'][i],
                'duration_sec': data['duration_sec'][i],
                'duration_frames': data['duration_frames'][i],
                'fps': [],
                'gpu_usage': [],
                'objects_per_frame': []
            }
        
        # Append current frame data to lists
        separated_data[video_path]['fps'].append(data['fps'][i])
        separated_data[video_path]['gpu_usage'].append(data['gpu_usage'][i])
        separated_data[video_path]['objects_per_frame'].append(data['objects_per_frame'][i])
    
    return separated_data

# Function to calculate mean values of the metrics
def calculate_mean_values(separated_data):
    for video_path, data in separated_data.items():
        # Calculate the mean of 'fps'
        data['mean_fps'] = sum(data['fps']) / len(data['fps']) if data['fps'] else 0
        
        # Calculate the mean of 'gpu_usage'
        data['mean_gpu_usage'] = sum(data['gpu_usage']) / len(data['gpu_usage']) if data['gpu_usage'] else 0
        
        # Calculate the mean of 'objects_per_frame'
        data['mean_objects_per_frame'] = sum(data['objects_per_frame']) / len(data['objects_per_frame']) if data['objects_per_frame'] else 0
        
        # Remove the original lists
        del data['fps'], data['gpu_usage'], data['objects_per_frame']
    
    return separated_data

# Function to create a DataFrame from the dictionary data
def create_df(data):
    df = pd.DataFrame(list(data.values()), index=data.keys())
    df['file_path'] = df.index
    df.reset_index(drop=True, inplace=True)
    return df

# Process all feeds
feed1_data = calculate_mean_values(separate_video_wise(feed1, 1))
feed2_data = calculate_mean_values(separate_video_wise(feed2, 2))
feed3_data = calculate_mean_values(separate_video_wise(feed3, 3))
feed4_data = calculate_mean_values(separate_video_wise(feed4, 4))

# Create DataFrames for all feeds
df_feed1 = create_df(feed1_data)
df_feed2 = create_df(feed2_data)
df_feed3 = create_df(feed3_data)
df_feed4 = create_df(feed4_data)

# Concatenate all feed DataFrames
df_feeds = pd.concat([df_feed1, df_feed2, df_feed3, df_feed4], ignore_index=True)

# Save the concatenated DataFrame to an Excel file
df_feeds.to_excel(r'results2.xlsx')

# Print the first few rows of the DataFrame
print(df_feeds.head())

# Summary statistics for the numerical columns
summary_stats = df_feeds.describe()
print(summary_stats)

# Summary of object detections across different feeds
object_detection_summary = df_feeds.groupby('feed_id').agg({
    'mean_objects_per_frame': 'mean',
    'mean_fps': 'mean',
    'mean_gpu_usage': 'mean'
}).reset_index()
print(object_detection_summary)

# Set up the matplotlib figure for distributions
plt.figure(figsize=(8, 4))

# Plot distribution of mean_fps
plt.subplot(1, 2, 1)
sns.histplot(df_feeds['mean_fps'], bins=10, kde=True)
plt.title('Distribution of Mean FPS')
plt.xlabel('Mean FPS')
plt.ylabel('Frequency')

# Plot distribution of mean_gpu_usage
plt.subplot(1, 2, 2)
sns.histplot(df_feeds['mean_gpu_usage'], bins=10, kde=True)
plt.title('Distribution of Mean GPU Usage')
plt.xlabel('Mean GPU Usage')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Correlation between mean_gpu_usage and mean_objects_per_frame
correlation = df_feeds['mean_gpu_usage'].corr(df_feeds['mean_objects_per_frame'])
print("Correlation:", correlation)

# Set up the matplotlib figure for scatter and box plots
plt.figure(figsize=(10, 5))

# Scatter plot of Mean Objects per Frame vs. Mean FPS
plt.subplot(1, 2, 1)
sns.scatterplot(x=df_feeds['mean_fps'], y=df_feeds['mean_objects_per_frame'])
plt.title('Mean Objects per Frame vs. Mean FPS')
plt.xlabel('Mean FPS')
plt.ylabel('Mean Objects per Frame')

# Box plot of Mean Objects per Frame for Each Feed ID
plt.subplot(1, 2, 2)
sns.boxplot(x=df_feeds['feed_id'], y=df_feeds['mean_objects_per_frame'])
plt.title('Mean Objects per Frame for Each Feed ID')
plt.xlabel('Feed ID')
plt.ylabel('Mean Objects per Frame')

plt.tight_layout()
plt.show()

# Set up the matplotlib figure for box plots of metrics
plt.figure(figsize=(12, 12))

# Define the list of metrics to visualize feed-wise
metrics = ['mean_fps', 'mean_gpu_usage', 'mean_objects_per_frame']
titles = ['Mean FPS per Feed', 'Mean GPU Usage per Feed', 'Mean Objects per Frame per Feed']
y_labels = ['Mean FPS', 'Mean GPU Usage', 'Mean Objects per Frame']

# Create a 3x1 subplot for visualizing each metric feed-wise
for i, metric in enumerate(metrics):
    plt.subplot(3, 1, i + 1)
    sns.boxplot(x=df_feeds['feed_id'], y=df_feeds[metric])
    plt.title(titles[i])
    plt.xlabel('Feed ID')
    plt.ylabel(y_labels[i])

plt.tight_layout()
plt.show()

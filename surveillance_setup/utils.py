import os

def get_video_paths(directory):
    video_paths = []
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.flv', '.webm']

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file_path)
            if ext.lower() in video_extensions:
                video_paths.append(file_path)

    return video_paths

def log_results(results_txt):
    with open('optimized_processing_logs.txt', 'a') as f:
        f.write(results_txt + '\n')
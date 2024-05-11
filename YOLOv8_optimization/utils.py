import os

def get_video_paths(directory):
    '''
    Traverse a directory recursively and return a list of full paths to video files.
    '''
    video_paths = []
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.flv', '.webm']

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file_path)
            if ext.lower() in video_extensions:
                video_paths.append(file_path)

    return video_paths

def log_results(results_txt, filename='optimized_processing_logs.txt'):
    '''
    This function writes a string of text to a log file
    '''
    with open(filename, 'a') as f:
        f.write(results_txt + '\n')
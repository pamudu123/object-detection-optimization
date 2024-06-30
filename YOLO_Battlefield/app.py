import cv2
import time
import os
import random
import json
import numpy as np
import pandas as pd
from ultralytics import YOLO
import gradio as gr

#############################
########### UTILS ###########
#############################

file_path = r'available_models.json'
with open(file_path, 'r') as file:
    yolo_data = json.load(file)

def extract_models_from_json():    
    model_names = []
    for version in yolo_data.values():
        for model in version:
            model_names.append(model["model"])
    
    return model_names

def extract_model_info(model_name):

    # Iterate through each YOLO version in the provided data
    for version, models in yolo_data.items():
        # Iterate through the models in each version
        for model in models:
            # Check if the model name matches the requested model
            if model['model'] == model_name:
                # Extract and return the relevant information
                return {
                    "model": model.get("model", None),
                    "size": model.get("size", None),
                    "mAPval_50-95": model.get("mAPval_50-95", None),
                    "params_M": model.get("params_M", None),
                    "FLOPs_B": model.get("FLOPs_B", None),
                    "mAPval_50": model.get("mAPval_50", None)
                }
            

def generate_html_table(data):
    html_content = """
    <table border="1" style="border-collapse: collapse; width: 90%; margin: 5px auto;">
    """
    for key, value in data.items():
        if value is not None:
            html_content += f"""
            <tr>
                <td style="padding: 6px; text-align: center;">{key}</td>
                <td style="padding: 6px; text-align: center;"><h5>{value}</h5></td>
            </tr>
            """
    html_content += "</table>"
    return html_content


def get_examples(directory):
    item_names = os.listdir(directory)
    paths = [os.path.join(directory, item) for item in item_names]
    return paths


#############################
##### PROCESS FUNCTIONS #####
#############################

def initialize_model(model_name: str):
    if 'yolov5' in model_name:
        model_name = model_name + "u"  # for ultralytics naming convention
    torch_model_name = f'{model_name}.pt'
    model = YOLO(torch_model_name)
    return model


def rearrange_detections_and_confidences(class_confidences, class_detections, class_dict, total_frames=1):
    class_confidences_ = {class_dict[class_id]: sum(confidences) / len(confidences) for class_id, confidences in class_confidences.items()}
    class_detections_ = {class_dict[class_id]: n_detections/total_frames for class_id, n_detections in class_detections.items()}

    class_confidences_ = dict(sorted(class_confidences_.items(), key=lambda x: x[0]))
    class_detections_ = dict(sorted(class_detections_.items(), key=lambda x: x[0]))

    df_class_confidences = pd.DataFrame(list(class_confidences_.items()), columns=['object', 'conf score'])
    df_class_detections = pd.DataFrame(list(class_detections_.items()), columns=['object', 'detections'])
    
    return df_class_confidences, df_class_detections


def process_detections(results, class_detections={}, class_confidences={}):
    for detection in results[0].boxes:
        class_id = int(detection.cls)
        confidence = float(detection.conf)

        if class_id in class_detections:
            class_detections[class_id] += 1
            class_confidences[class_id].append(confidence)
        else:
            class_detections[class_id] = 1
            class_confidences[class_id] = [confidence]

    return class_detections, class_confidences

def add_fps_to_frame(frame, fps):
    # Display FPS on the top-left corner
    frame[:30, :150] = (0, 0, 0)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def add_model_name_to_frame(frame, model_name):
    # Display model name in the bottom-right corner
    h, w, _ = frame.shape
    frame[h-30:h, w-120:w] = (0, 0, 0)
    cv2.putText(frame, model_name, (w-110, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def calculate_fps(t_prev, t_new):
    return 1 / (t_new - t_prev)

def generate_bar_plot(data, x, y, title, tooltip, y_lim):
    return gr.BarPlot(
        data, x=x, y=y,
        title=title,
        tooltip=tooltip,
        y_lim=y_lim,
    )


#############################
####### PROCESS IMAGE #######
#############################

def process_image(np_image: np.ndarray, model_name: str, conf: float = 0.25, iou: float = 0.5, img_size: int = 640, device: str = 'cpu'):
    print(time.ctime())
    if (np_image is None) or (np_image.size == 0):
        return None, None, None, None

    model = initialize_model(model_name)

    class_dict = model.names
    t_start = time.time()

    results = model(np_image, imgsz=img_size, conf=conf, iou=iou, half=False, device=device, verbose=False)

    class_detections, class_confidences = process_detections(results)

    # add annotations
    annotated_image = results[0].plot()
    annotated_image = add_model_name_to_frame(annotated_image, model_name)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # save
    save_dir = r'tmp_images'
    os.makedirs(save_dir, exist_ok=True)
    output_path = f'{save_dir}/{model_name}_{len(os.listdir(save_dir))+1}.jpg'
    cv2.imwrite(output_path, annotated_image)

    processing_time = time.time() - t_start
    processing_time = f'{round(processing_time*1000, 2)} ms/frame'

    # rearrange
    df_class_confidences, df_class_detections = rearrange_detections_and_confidences(class_confidences, class_detections, class_dict)

    confidence_barplot = generate_bar_plot(df_class_confidences, "object", "conf score", "Distribution of Class Confidences", ["object", "conf score"], [0, 1])
    detection_barplot = generate_bar_plot(df_class_detections, "object", "detections", "Distribution of Class Detections", ["object", "detections"], [0, 20])

    return output_path, processing_time, confidence_barplot, detection_barplot


#############################
####### PROCESS VIDEO #######
#############################

def process_video(video_path, model_name, frame_limit, conf: float = 0.25, iou: float = 0.5, img_size: int = 640, device: str = 'cpu'):
    print(time.ctime())
    if (video_path is None) or len(video_path) < 2:
        return None, None, None, None, None

    model = initialize_model(model_name)

    t_prev = t_start = time.time()
    fps_sum = 0.0
    frame_count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    class_dict = model.names

    # setup video saving
    cap = cv2.VideoCapture(video_path)
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frame_limit)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    save_dir = r'tmp_videos'
    os.makedirs(save_dir, exist_ok=True)
    output_path = f'{save_dir}/{os.path.basename(video_path).split(".")[0]}__{model_name}_{random.randint(1000, 9999)}.mp4'
    video_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))

    # Initialize detection and confidence tracking
    class_detections = {}
    class_confidences = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count >= frame_limit:
            break

        results = model(frame, imgsz=img_size, conf=conf, iou=iou, half=False, device=device, verbose=False)

        t_new = time.time()
        fps = calculate_fps(t_prev, t_new)
        t_prev = t_new

        fps_sum += fps
        frame_count += 1

        class_detections, class_confidences = process_detections(results, class_detections, class_confidences)

        annotated_frame = results[0].plot()
        annotated_frame = add_fps_to_frame(annotated_frame, fps)
        annotated_frame = add_model_name_to_frame(annotated_frame, model_name)

        video_out.write(annotated_frame)

    avg_fps = fps_sum / frame_count
    avg_frame_processing_time = (time.time() - t_start) / total_frames

    # rearrange
    df_class_confidences, df_class_detections = rearrange_detections_and_confidences(class_confidences, class_detections, class_dict, total_frames)

    confidence_barplot = generate_bar_plot(df_class_confidences, "object", "conf score", "Distribution of Class Confidences", ["object", "conf score"], [0, 1])
    detection_barplot = generate_bar_plot(df_class_detections, "object", "detections", "Distribution of Class Detections", ["object", "detections"], [0, 10])

    avg_frame_processing_time = f'{round(avg_frame_processing_time*1000, 2)} ms/frame'
    
    return output_path, round(avg_fps, 4), avg_frame_processing_time, confidence_barplot, detection_barplot


#############################
########  GRADIO APP ########
#############################
def reset_sliders():
    return gr.update(value=0.25), gr.update(value=0.5), gr.update(value=640),

def reset_display_widgets():
    return gr.update(value=None), gr.update(value=None), gr.update(value=None),

def clear_image_tab():
    return (
        *reset_display_widgets(),
        *reset_sliders(),
        gr.update(value=None), gr.update(value=None),
        gr.update(value=None), gr.update(value=None),
        gr.update(value=None), gr.update(value=None)
    )

def clear_video_tab():
    return (
        *reset_display_widgets(),
        *reset_sliders(),
        gr.update(value=None), gr.update(value=None),
        gr.update(value=None), gr.update(value=None),
        gr.update(value=None), gr.update(value=None),
        gr.update(value=None), gr.update(value=None)
    )

def update_model_info(model_name):
    return generate_html_table(extract_model_info(model_name))

def create_video_tab(available_models):
    with gr.Tab("Video"):        
        with gr.Accordion("Detection Parameters", open=False):
            confidence_slider = gr.Slider(label="Confidence", minimum=0, maximum=1, step=0.01, value=0.25, interactive=True)
            iou_slider = gr.Slider(label="IOU", minimum=0, maximum=1, step=0.01, value=0.5, interactive=True)
            image_size_slider = gr.Slider(label="Image Size", minimum=32, maximum=1280, step=32, value=640, interactive=True)

        with gr.Accordion("Other Parameters", open=False):
            device_radio = gr.Radio(choices=["cpu", "cuda"], label="Device", value='cpu', interactive=False)
            nframes_slider = gr.Slider(label="Frames to Process", minimum=1, maximum=100, step=1, value=80, info="Process only the first #n frames of the video (Due to compute limitation)", interactive=True)

        with gr.Row():
            gr.Markdown("")
            model_dropdown1 = gr.Dropdown(choices=available_models, label="Model 1", value=available_models[1])
            model_dropdown2 = gr.Dropdown(choices=available_models, label="Model 2", value=available_models[5])

        with gr.Row():
            gr.Markdown("")
            model_info1 = gr.HTML(show_label=False, value=update_model_info(model_dropdown1.value))
            model_info2 = gr.HTML(show_label=False, value=update_model_info(model_dropdown2.value))

        with gr.Row():
            video_input = gr.Video(label="Upload Video")
            video_output1 = gr.Video(label="Model1 Output")
            video_output2 = gr.Video(label="Model2 Output")

        with gr.Row():
            clear_button = gr.Button("Clear")
            process_button1 = gr.Button("Run Model 1")
            process_button2 = gr.Button("Run Model 2")

        with gr.Row():
            gr.Markdown("## Average FPS")
            avg_fps1 = gr.Textbox(show_label=False, container=False, text_align='right')
            avg_fps2 = gr.Textbox(show_label=False, container=False, text_align='right')

        with gr.Row():
            gr.Markdown("## Average Processing Time")
            avg_frame_processing_time1 = gr.Textbox(show_label=False, container=False, text_align='right')
            avg_frame_processing_time2 = gr.Textbox(show_label=False, container=False, text_align='right')

        with gr.Row():
            conf_description = """
## Confidence Scores
This plot shows the average confidence scores for each detected object across all processed frames. The confidence score indicates how certain the system is about the detection of each object.
"""
            gr.Markdown(conf_description)
            conf_plot1 = gr.BarPlot()
            conf_plot2 = gr.BarPlot()

        with gr.Row():
            detection_description = """
## Total Detections Per Frame
This plot displays the total number of detections for each object. The total detections are divided by the number of processed frames to show the average number of objects detected per frame.
"""
            gr.Markdown(detection_description)
            detection_plot1 = gr.BarPlot()
            detection_plot2 = gr.BarPlot()

        process_button1.click(
            process_video,
            inputs=[video_input, model_dropdown1, nframes_slider, confidence_slider, iou_slider, image_size_slider, device_radio],
            outputs=[video_output1, avg_fps1, avg_frame_processing_time1, conf_plot1, detection_plot1],
        )

        process_button2.click(
            process_video,
            inputs=[video_input, model_dropdown2, nframes_slider, confidence_slider, iou_slider, image_size_slider, device_radio],
            outputs=[video_output2, avg_fps2, avg_frame_processing_time2, conf_plot2, detection_plot2],
        )

        clear_button.click(
            fn=clear_video_tab,
            outputs=[video_input, video_output1, video_output2, confidence_slider, iou_slider, image_size_slider,
                     avg_fps1, avg_fps2,
                     avg_frame_processing_time1, avg_frame_processing_time2, 
                     conf_plot1, conf_plot2, detection_plot1, detection_plot2]
        )

        gr.Examples(examples=get_examples(r'examples/videos'), inputs=video_input)

        model_dropdown1.change(update_model_info, inputs=model_dropdown1, outputs=model_info1)
        model_dropdown2.change(update_model_info, inputs=model_dropdown2, outputs=model_info2)


def create_image_tab(available_models):
    with gr.Tab("Image"):        
        with gr.Accordion("Detection Parameters", open=False):
            confidence_slider = gr.Slider(label="Confidence", minimum=0, maximum=1, step=0.01, value=0.25, interactive=True)
            iou_slider = gr.Slider(label="IOU", minimum=0, maximum=1, step=0.01, value=0.5, interactive=True)
            image_size_slider = gr.Slider(label="Image Size", minimum=32, maximum=1280, step=32, value=640, interactive=True)

        with gr.Accordion("Other Parameters", open=False):
            device_radio = gr.Radio(choices=["cpu", "cuda"], label="Device", value='cpu', interactive=False)

        with gr.Row():
            gr.Markdown("")
            model_dropdown1 = gr.Dropdown(choices=available_models, label="Model 1", value=available_models[1])
            model_dropdown2 = gr.Dropdown(choices=available_models, label="Model 2", value=available_models[5])

        with gr.Row():
            gr.Markdown("")
            model_info1 = gr.HTML(show_label=False, value=update_model_info(model_dropdown1.value))
            model_info2 = gr.HTML(show_label=False, value=update_model_info(model_dropdown2.value))

        with gr.Row():
            image_input = gr.Image(label="Upload Image")
            image_output1 = gr.Image(label="Model1 Output")
            image_output2 = gr.Image(label="Model2 Output")

        with gr.Row():
            clear_button = gr.Button("Clear")
            process_button1 = gr.Button("Run Model 1")
            process_button2 = gr.Button("Run Model 2")

        with gr.Row():
            gr.Markdown("## Processing Time")
            avg_frame_processing_time1 = gr.Textbox(show_label=False, container=False, text_align='right')
            avg_frame_processing_time2 = gr.Textbox(show_label=False, container=False, text_align='right')

        with gr.Row():
            conf_description = """
## Confidence Scores
This plot shows the average confidence scores for each detected object in the processed image. The confidence score indicates how certain the system is about the detection of each object.
"""
            gr.Markdown(conf_description)
            conf_plot1 = gr.BarPlot()
            conf_plot2 = gr.BarPlot()

        with gr.Row():
            detection_description = """
## Total Detections
This plot displays the total number of detections for each object in the processed image.
"""
            gr.Markdown(detection_description)
            detection_plot1 = gr.BarPlot()
            detection_plot2 = gr.BarPlot()

        process_button1.click(
            process_image,
            inputs=[image_input, model_dropdown1, confidence_slider, iou_slider, image_size_slider, device_radio],
            outputs=[image_output1, avg_frame_processing_time1, conf_plot1, detection_plot1],
        )

        process_button2.click(
            process_image,
            inputs=[image_input, model_dropdown2, confidence_slider, iou_slider, image_size_slider, device_radio],
            outputs=[image_output2, avg_frame_processing_time2, conf_plot2, detection_plot2],
        )

        clear_button.click(
            fn=clear_image_tab,
            outputs=[image_input, image_output1, image_output2, confidence_slider, iou_slider, image_size_slider,
                     avg_frame_processing_time1, avg_frame_processing_time2, conf_plot1, conf_plot2, detection_plot1, detection_plot2]
        )

        gr.Examples(examples=get_examples(r'examples/images'), inputs=image_input)

        model_dropdown1.change(update_model_info, inputs=model_dropdown1, outputs=model_info1)
        model_dropdown2.change(update_model_info, inputs=model_dropdown2, outputs=model_info2)

# Main function to create the Gradio interface
def main():
    available_models = extract_models_from_json()
    print(available_models)
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# YOLO Battlefield")

        ## DESCRIPTION ON THE SPACE
        space_description = """
> **Note** \\
> This space is currently running on a CPU instance, so processing times are high, and FPS values are comparatively low compared to GPU. However, this does not affect the relative comparison between the models.
>
> Sometimes, when you run a model for the first time, the processing time might be inaccurate. If this happens, rerun with the same configurations to get more accurate processing time.

**Upcoming Features:**
- Integration of more YOLO models

**Feedback and Suggestions:**
Please contact  [Pamudu Ranasinghe](https://www.linkedin.com/in/pamudu-ranasinghe/)
"""
        ### Try Battling with YOLO Models
        gr.Markdown(space_description)
        create_image_tab(available_models)
        create_video_tab(available_models)

    demo.launch()

if __name__ == "__main__":
    main()

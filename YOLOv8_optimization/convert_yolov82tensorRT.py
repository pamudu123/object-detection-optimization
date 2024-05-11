## TensorRT conversion
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.export(
    format="engine",
    imgsz=640,
    dynamic=True,
    verbose=True,
    batch=8,
    workspace=2,
    int8=True,
    data="data.yaml"  # COCO, ImageNet, or DOTAv1 for appropriate model task
)

tensorrt_model = YOLO('yolov8n.engine')
results = tensorrt_model('https://ultralytics.com/images/bus.jpg')
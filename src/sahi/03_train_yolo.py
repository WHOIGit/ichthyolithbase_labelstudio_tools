from ultralytics import YOLO



# Load a pretrained YOLO11n model
yolo_model_name = "yolo11n"
yolo_model_name = "yolo11s"
imgsz = 640
project_name = f'runs/slice_train_output/{yolo_model_name}__imgsz-{imgsz}__slice-1024-33__fuecocoFIXED4'

common_kwargs = dict(
    epochs=500,  # Number of training epochs
    imgsz=imgsz,  # Image size for training
    device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    patience=50,
    save=True,
    project=project_name,
    single_cls = False,
)

model = YOLO(f'{yolo_model_name}')
dataset_yaml = 'datasets/yolo/fuecoco_FIXED4__s1024_o33__box/data.yml'

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data=dataset_yaml,  # Path to dataset configuration file
    name='detect',
    **common_kwargs,
)




model_seg = YOLO(f'{yolo_model_name}-seg.pt')
dataset_yaml_seg = 'datasets/yolo/fuecoco_FIXED4__s1024_o33__seg/data.yml'

train_results_seg = model_seg.train(
    data=dataset_yaml_seg,  # Path to dataset configuration file
    name='segment',
    **common_kwargs,
)

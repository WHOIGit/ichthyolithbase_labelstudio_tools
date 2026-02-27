import json
import os
import pickle
from PIL import Image
Image.MAX_IMAGE_PIXELS = 400_000_000  # 20k * 20k pixels


#from sahi.predict import predict
from patch_sahi import predict


# implement this later if you want to compare temp
# seg_labels = ['tooth','denticle','other','fragment','unsure']
#
# coco_dict = dict(images=[], annotations=[], info={}, licenses=[],
#     categories = [dict(id=i,name=label) for i,label in enumerate(seg_labels)])
# for idx,img in enumerate(sorted(os.listdir(img_dir))):
#     img_path = os.path.join(img_dir,img)
#     width,height = Image.open(img_path).size
#     coco_image = dict(file_name = img,
#                       id = idx,
#                       width = width,
#                       height = height)
#     coco_dict['images'].append(coco_image)

def do_that_thing(model_path, img_dir, save_dir, segment=True):

    name = 'segment' if segment else 'detect'
    if segment: assert 'segment' in model_path
    else: assert 'detect' in model_path

    results = predict(
        model_type="ultralytics",
        model_path=model_path,
        model_device="cuda:0",  # or 'cuda:0'
        model_confidence_threshold=0.3,
        source=img_dir,
        image_size = 1024,

        slice_height=1024,
        slice_width=1024,
        overlap_height_ratio=0.333,
        overlap_width_ratio=0.333,
        postprocess_type = 'NMM',
        postprocess_match_metric = 'IOS',
        postprocess_class_agnostic = True,
        #postprocess_match_threshold = 0.66,
        force_postprocess_type = False,

        project = save_dir,
        name = name,
        export_crop = False,
        #export_pickle = True,
        #dataset_json_path = coco_dict,
        #exclude_classes_by_name = ['unsure'],

        visual_bbox_thickness = 2,
        visual_text_size = 1,
        visual_text_thickness= 2,
        visual_colors=['008000', '0000ff', 'ffa500', '800080', 'ff0000'],
        visual_label_pattern= "{OBJID} {CONF:.2f}", #"{OBJID} {LABEL:.1} {CONF:.2f}",
        visual_export_format = 'jpg',

        return_dict=True,
    )
    return results  # output_dir:str, coco:CocoPlus


#model_path = 'runs/slice_train_output/segment/yolo11n_slice-1024-33/weights/best.pt'
#model_path = 'runs/slice_train_output/detect/yolo11n_slice-1024-33/weights/best.pt'

model_path = 'runs/slice_train_output/yolo11s__imgsz-640__slice-1024-33__fuecocoFIXED4/detect/weights/best.pt'
model_path = 'runs/slice_train_output/yolo11s__imgsz-640__slice-1024-33__fuecocoFIXED4/segment/weights/best.pt'

model_path = 'runs/slice_train_output/yolo11s__imgsz-640__slice-1024-33__fuecocoFIXED4/detect2/weights/best.pt'
model_path = 'runs/slice_train_output/yolo11s__imgsz-640__slice-1024-33__fuecocoFIXED4/segment2/weights/best.pt'

img_dir = 'datasets/intermediary/fullframe_halfsize_jpg' #/121_U1553D_4R_6W_16-19cm__N1of2_Z200.jpg'

x = do_that_thing(model_path, img_dir, 'runs/inference_sahi', segment=True)

with open(f"{x['export_dir']}/predictions.coco.json", 'w') as f:
    json.dump(x['coco'].json, f)

print(x)



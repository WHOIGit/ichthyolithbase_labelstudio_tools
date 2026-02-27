import os
import copy
import json
import shutil
from pathlib import Path

import sahi.slicing
import sahi.utils.coco

from patch_sahi import export_single_yolo_image_and_corresponding_txt_seg, \
    export_single_yolo_image_and_corresponding_txt_box, \
    slice_coco


def coco_to_yolo(coco, output_dir, image_dir='.',
                 train_split_ratio=0.8, numpy_seed=0, disable_symlink=False,
                 export_segments=False):
    if isinstance(coco,sahi.utils.coco.Coco):
        pass
    else:
        coco = sahi.utils.coco.Coco.from_coco_dict_or_path(coco)  # annotations only
        coco.image_dir = image_dir


    if export_segments:
        # this overwrites the annotation reading+export function
        sahi.utils.coco.export_single_yolo_image_and_corresponding_txt = export_single_yolo_image_and_corresponding_txt_seg
    else:
        # this is supposed to reset the function to the original one
        sahi.utils.coco.export_single_yolo_image_and_corresponding_txt = export_single_yolo_image_and_corresponding_txt_box

    yaml_path = sahi.utils.coco.export_coco_as_yolo(
        output_dir=output_dir,
        train_coco=coco,
        train_split_rate=train_split_ratio,
        numpy_seed=numpy_seed,
        disable_symlink=disable_symlink,
    )
    return yaml_path


if __name__ == '__main__':
    coco_obj, original_size_mapper = slice_coco(
        coco_dict_or_path='datasets/annotations/fuecoco_FIXED4.json',
        image_dir='.',
        output_dir='datasets/intermediary/sliced__s1024_o33',
        slice_height=1024,
        slice_width=1024,
        overlap_height_ratio=0.333,
        overlap_width_ratio=0.333,
        min_area_ratio=0.05,
        out_ext='.jpg')
    with open('datasets/annotations/fuecoco_FIXED4__sliced_1024_33.json','w') as f:
        json.dump(coco_obj.json, f, indent=2)

    box_yaml = coco_to_yolo(coco_obj, 'datasets/yolo/fuecoco_FIXED4__s1024_o33__box',
                            image_dir='datasets/intermediary/sliced__s1024_o33', export_segments=False)
    seg_yaml = coco_to_yolo(coco_obj, 'datasets/yolo/fuecoco_FIXED4__s1024_o33__seg',
        image_dir='datasets/intermediary/sliced__s1024_o33', export_segments=True)

    print(seg_yaml)
    print(box_yaml)





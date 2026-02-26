import json
import os

from sahi.utils.coco import Coco, CocoAnnotation

from core import LabelStudioPlus

lsp = LabelStudioPlus.from_config('configs/ls_config.ichthyolith_sahi.json')
lsp.set_project('Fullframe Imagery')

tasks = lsp.get_tasks(limit_fields_to=['id','data','annotations'])
tasks_by_pk = {t['data']['fullframe_id']:t for t in tasks}

fuecoco = Coco.from_coco_dict_or_path('datasets/annotations/fuecoco_121N1of2.json')
categories_by_name = {cat.name:cat for cat in fuecoco.categories}
coco_image_by_filenamestem = {os.path.splitext(os.path.basename(img.file_name))[0]: img for img in fuecoco.images}
from pprint import pprint

for pk,cocoimg in coco_image_by_filenamestem.items():
    task = tasks_by_pk[pk]
    # we are now correcting the coco annotations with the task annotations
    coco_annots_by_bbox = {tuple(map(int,ann.bbox)):ann for ann in cocoimg.annotations}

    task_annots_by_bbox = {}
    for ann in task['annotations'][0]['result']:
        # these are in percents of the whole image
        box = (ann['value']['x'],
               ann['value']['y'],
               ann['value']['width'],
               ann['value']['height'])
        cocobox = (box[0]*ann['original_width']/100+0.5,
                   box[1]*ann['original_height']/100+0.5,
                   box[2]*ann['original_width']/100+0.5,
                   box[3]*ann['original_height']/100+0.5)
        cocobox = tuple(map(int, cocobox))
        task_annots_by_bbox[cocobox] = ann['value']

    task_bboxes = set(task_annots_by_bbox.keys())
    coco_bboxes = set(coco_annots_by_bbox.keys())
    new_annotations_bboxes = task_bboxes-coco_bboxes
    removed_annotations_bboxes = coco_bboxes-task_bboxes
    relabeled_annotations_bboxes = [box for box in task_bboxes.intersection(coco_bboxes) if task_annots_by_bbox[box]['rectanglelabels'][0].lower() != coco_annots_by_bbox[box].category_name ]
    print(f"{os.path.basename(cocoimg.file_name)}: old_total: {len(coco_bboxes)}, new_total: {len(task_bboxes)}, new_annots: {len(new_annotations_bboxes)}, removed_annots: {len(removed_annotations_bboxes)}, relabeled: {len(relabeled_annotations_bboxes)}")

    # removing entries
    for box in removed_annotations_bboxes:
        coco_annots_by_bbox.pop(box)

    # relabeling entries
    for box in relabeled_annotations_bboxes:
        updated_label = task_annots_by_bbox[box]['rectanglelabels'][0].lower()
        coco_annots_by_bbox[box].category_name = updated_label
        coco_annots_by_bbox[box].category_id = categories_by_name[updated_label].id

    # new entries
    for box in new_annotations_bboxes:
        new_label = task_annots_by_bbox[box]['rectanglelabels'][0].lower()
        coco_annots_by_bbox[box] = CocoAnnotation(
            category_id = categories_by_name[new_label].id,
            category_name = new_label,
            image_id = cocoimg.id,
            bbox = list(box),
            # we dont have segmentations, so when this gets sliced, it'll be a bbox not a segmentation
        )

    cocoimg.annotations = list(coco_annots_by_bbox.values())

with open('datasets/annotations/fuecoco_121N1of2_FIXED.json', 'w') as f:
    json.dump(fuecoco.json, f)

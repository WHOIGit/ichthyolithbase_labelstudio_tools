import os
import json
import math
from copy import deepcopy

import numpy as np
from sahi.annotation import BoundingBox
from sahi.utils.coco import Coco, CocoImage, CocoAnnotation
from tqdm import tqdm
from PIL import Image

from labelstudio_tools.utils import simple_task_filter_builder
from proj_utils import sort_prediction_result_SCANLINE

Image.MAX_IMAGE_PIXELS = 20_000*20_000

import label_studio_sdk.converter as ls_converter
from label_studio_sdk.converter.exports.brush_to_coco import generate_contour_from_rle

from sahi.prediction import ObjectPrediction, PredictionResult
#from sahi.utils.cv import visualize_object_predictions
from sahi.postprocess.combine import NMMPostprocess

from labelstudio_tools import LabelStudioPlus
from patch_sahi import visualize_object_predictions

def bitmask2img(bitmask, outfile):
  if bitmask.max() <= 1:
    bitmask = bitmask*255
  img = Image.fromarray(bitmask).convert('L').convert('1')
  img.save(outfile)
  return img


def download_ls_tasks_and_images(lsp,
                                 label_group:str,
                                 key_fields='fullframe_id',
                                 save_dir = 'datasets/fullframe',
                                 filter_dict=None,
                                 fullframe=('fullframe_id','jpg'),
                                 fullframe_fullres=None):

    seg_labels = lsp.config_control_labels()[label_group]
    seg_label_colors = []
    for seg_label in seg_labels:
        label_color = lsp.config_control_labels_detailed()[label_group][seg_label].attr['background']
        seg_label_colors.append(label_color)
    tasks = lsp.get_tasks(limit_fields_to=['id','data','annotations'], filter_dict=filter_dict)

    data = {}
    for task in tqdm(tasks):
        fullframe_key = lsp.task_datafields_key(task, key_fields)

        if fullframe_key not in data:
            # download fullframe images
            datafield, ext = fullframe
            assert datafield in task['data']
            os.makedirs(save_dir, exist_ok=True)
            fullframe_localfile = os.path.join(save_dir, f"{fullframe_key}.{ext}")
            lsp.download_s3url(task['data'][datafield], fullframe_localfile, clobber=False)
            fullframe_shape = Image.open(fullframe_localfile).size  # (width, height)
            fullframe_shape = dict(w=fullframe_shape[0],h=fullframe_shape[1])

            data[fullframe_key] = dict(
                slices=[], bitmasks=[], sahi_preds=[],
                fullframe_shape = fullframe_shape,
                fullframe_localpath = fullframe_localfile,
            )
            if fullframe_fullres:
                datafield, ext = fullframe_fullres
                savedir_fullres = save_dir + '_fullres' if not save_dir.endswith('/') else save_dir[:-1] + '_fullres'
                assert datafield in task['data']
                os.makedirs(savedir_fullres, exist_ok=True)
                fullframe_fullres_localfile = os.path.join(savedir_fullres, f"{fullframe_key}.{ext}")
                lsp.download_s3url(task['data'][datafield], fullframe_fullres_localfile, clobber=False)
                fullframe_fullres_shape = Image.open(fullframe_fullres_localfile).size  # (width, height)
                data[fullframe_key]['fullframe_fullres_shape'] = dict(w=fullframe_fullres_shape[0],h=fullframe_fullres_shape[1])
                data[fullframe_key]['fullframe_fullres_localpath'] = fullframe_fullres_localfile
        else:
            fullframe_shape = data[fullframe_key]['fullframe_shape']

        # get slice annotations
        slice_data = {}
        annotation_results = [r for r in task['annotations'][0]['result'] if 'brushlabels' in r['value']]
        slice_data['masks_rle'] = [r['value']['rle'] for r in annotation_results]
        slice_data['mask_labels'] = [r['value']['brushlabels'][0] for r in annotation_results]
        slice_data['mask_scores'] = [r['score'] for r in annotation_results]
        offsets = task['data']['slice_offset']  # todo else get it from slice filename
        slice_shape =  task['data']['shape']
        slice_data['offset'] = (offsets['x'],offsets['y'])
        slice_width,slice_height = (slice_shape['w'],slice_shape['h'])

        slice_data['bitmasks'] = []
        for i, rle in enumerate(slice_data['masks_rle']):
            mask_1d = ls_converter.brush.decode_rle(rle, print_params=False)
            bitmask = np.reshape(mask_1d, [slice_height, slice_width, 4])[:, :, 0]
            slice_data['bitmasks'].append(bitmask)

            seg_label = slice_data['mask_labels'][i]  # todo check this?
            score = slice_data['mask_scores'][i]

            seg, bboxes, areas = generate_contour_from_rle(rle, slice_width, slice_height)
            sahi_pred = ObjectPrediction(segmentation=seg,
                                         category_id=seg_labels.index(seg_label),
                                         category_name=seg_label,
                                         score=score,
                                         shift_amount=slice_data['offset'],
                                         full_shape=[fullframe_shape['h'], # height
                                                     fullframe_shape['w']], # width
                                         ).get_shifted_object_prediction()
            data[fullframe_key]['sahi_preds'].append(sahi_pred)
        data[fullframe_key]['slices'].append(slice_data)
    return tasks, seg_labels, seg_label_colors, data


def export_review_image(record, save_dir, multimasks_only=False, hide_mask=False, ext=None):
    result_img,ext_fflocal = os.path.splitext(os.path.basename(record['fullframe_localpath']))
    ext = ext or ext_fflocal[1:]
    annots = []
    if multimasks_only:
        for object_prediction in record['prediction_result'].object_prediction_list:
            if len(object_prediction.mask.segmentation)>1:
                annots.append(object_prediction)
                print(os.path.basename(record['fullframe_localpath']),
                      [len(s) for s in object_prediction.mask.segmentation])
    else:
        annots = record['prediction_result'].object_prediction_list

    if multimasks_only and len(annots)==0:
        return

    os.makedirs(save_dir, exist_ok=True)
    visualize_object_predictions(
      image=np.ascontiguousarray(record['prediction_result'].image),
      object_prediction_list=annots,
      rect_th=2,
      text_size=1,
      text_th=2,
      colors=['008000', '0000ff', 'ffa500', '800080', 'ff0000'],
      label_pattern="{OBJID} {LABEL:.1}",
      hide_mask = hide_mask,
      output_dir=save_dir,
      file_name=result_img,
      export_format=ext,
    )


def cleanup_masks(record, threshold=0.1):
    for object_prediction in record['prediction_result'].object_prediction_list:
        if len(object_prediction.mask.segmentation) > 1:
            subsegments = sorted(object_prediction.mask.segmentation, key=lambda s:len(s), reverse=True)
            kept_subsegs = [ subsegments[0] ]
            sebseg_lengths = [len(s) for s in subsegments]
            largest = sebseg_lengths[0]
            for subseg in subsegments[1:]:
                if len(subseg) >= largest*threshold:
                    kept_subsegs.append(subseg)
            print(f"{os.path.basename(record['fullframe_localpath'])}: {sebseg_lengths} -> {[len(s) for s in kept_subsegs]}")
            object_prediction.mask.segmentation = kept_subsegs


def stitch_annotation_slices(data, inplace=False):
    if not inplace:
        data = deepcopy(data)

    for record in tqdm(list(data.values())):
        NMM = NMMPostprocess(match_metric='IOS', match_threshold=0.5)  # IOS is important here, threshold variable.
                                                        # large objects spanning multiple slices like lower thresholds
        nmm_preds = NMM(record['sahi_preds'])
        prediction_result = PredictionResult(nmm_preds, record['fullframe_localpath'])
        record['prediction_result'] = prediction_result
        cleanup_masks(record, threshold=0.1)

        for obj in record['prediction_result'].object_prediction_list:
            obj.bbox = obj.bbox.get_expanded_box(0.1, max_x=prediction_result.image_width, max_y=prediction_result.image_height)

        #record['prediction_result'].object_prediction_list.sort(key=lambda obj: obj.bbox.box)
        #sort_prediction_result_CLUSTER(record['prediction_result'], eps=0.30)
        record['prediction_result'].object_prediction_list = sort_prediction_result_SCANLINE(record['prediction_result'].object_prediction_list)
    return data


def visualize_stitched_annotations(data,
    review_savedir='datasets/annotations/fullframe_review',
    review_savedir_multimask='datasets/annotations/fullframe_review_multimaskOnly', ext='jpg'):
    for key, record in tqdm(data.items()):
        export_review_image(record,
                            save_dir=review_savedir_multimask,
                            multimasks_only=True, ext=ext)
        export_review_image(record,
                            save_dir=review_savedir,
                            multimasks_only=False,
                            hide_mask=True, ext=ext)


def stitched_annotations_to_coco(data, labels):
    fuecoco = Coco(name='fuecoco')
    fuecoco.add_categories_from_coco_category_list([dict(id=i, name=label) for i, label in enumerate(labels)])
    for key, record in tqdm(data.items()):
        coco_image = dict(file_name=record['fullframe_localpath'],
                          width=record['fullframe_shape']['w'],
                          height=record['fullframe_shape']['h'])
        coco_image = CocoImage(**coco_image)

        for object_prediction in record['prediction_result'].object_prediction_list:
            coco_image.add_annotation(object_prediction.to_coco_prediction(image_id=coco_image.id))
        fuecoco.add_image(coco_image)
    return fuecoco


def export_coco(coco_dataset:Coco, outfile):
    print(coco_dataset.stats) # todo set this as coco.info
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'w') as f:
        json.dump(coco_dataset.json, f, indent=2)

def scale_to_fullres(data, inplace=False):
    if not inplace:
        data = deepcopy(data)

    for record in tqdm(list(data.values())):

        x_scale = record['fullframe_fullres_shape']['w'] / record['fullframe_shape']['w']
        y_scale = record['fullframe_fullres_shape']['h'] / record['fullframe_shape']['h']
        old_object_prediction_list:list[ObjectPrediction] = record['prediction_result'].object_prediction_list
        new_object_prediction_list = []

        for obj in old_object_prediction_list:
            if obj.mask:
                scaled_segmentation = deepcopy(obj.mask.segmentation)
                for idx, poly in enumerate(obj.mask.segmentation):
                    for i in range(0, len(poly), 2):
                        poly[i] *= x_scale
                        poly[i + 1] *= y_scale
                    scaled_segmentation[idx] = [int(px + 0.5) for px in poly]
                anno_kwarg = dict(segmentation=scaled_segmentation)
                # bbox for mask will be auto-calculated
            else:
                scaled_box = obj.bbox.box
                scaled_box[0] *= x_scale
                scaled_box[1] *= y_scale
                scaled_box[2] *= x_scale
                scaled_box[3] *= y_scale
                scaled_box = [int(px + 0.5) for px in scaled_box]
                anno_kwarg = dict(bbox=scaled_box)

            new_obj = ObjectPrediction(
                category_id = obj.category.id,
                category_name = obj.category.name,
                score = obj.score,
                full_shape = [record['fullframe_fullres_shape']['h'],
                              record['fullframe_fullres_shape']['w']],
                **anno_kwarg,
            )
            new_object_prediction_list.append(new_obj)

        record['prediction_result'] = PredictionResult(new_object_prediction_list, image=record['fullframe_fullres_localpath'])
    return data

if __name__ == '__main__':
    lsp = LabelStudioPlus.from_config('configs/ls_config.ichthyolith_sahi.json')
    filter_dict = simple_task_filter_builder('fullframe_id','121_U1553D_4R_6W_16-19cm__N1of2_Z200','equal')
    #filter_dict = None
    tasks, labels, label_colors, data = download_ls_tasks_and_images(
        lsp, 'tag',
        key_fields='fullframe_id',
        save_dir='datasets/s3cache/fullframe',
        filter_dict=filter_dict,
        fullframe=('fullframe__s3url', 'jpg'),
        #fullframe_fullres=('fullframe_fullres__s3url', 'tif')
    )
    stitch_annotation_slices(data, inplace=True)
    #scale_to_fullres(data, inplace=True)
    visualize_stitched_annotations(data,
        review_savedir = 'datasets/annotations/fullframe_review',
        review_savedir_multimask = 'datasets/annotations/fullframe_review_multimaskOnly',
        ext = 'jpg',
        )
    fuecoco = stitched_annotations_to_coco(data, labels)
    export_coco(fuecoco, 'datasets/annotations/fuecoco_121N1of2.json')





# TODO k send output fullframes to Elizabeth.
#      k enable saving visualizations as fullsize tiff
#      k upload bboxes to labelstudio
#      k make corrections from Elizabeth
#      k train model
#      k apply model to 121_U15... for more denticles
#      validate with elizabeth
#      train model
#      evaluate output of model inference with Elizabeth on untrained images

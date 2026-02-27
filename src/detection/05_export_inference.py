

# see bbox2roi

# output1: ROIs from the fullframe_fullres imagery

# output2: slice imagery and upload to LS as pre-annotations

# output2: bounding boxes, possible masks, to fullframe imagery
import json
import os
from tqdm import tqdm
from pathlib import Path
from typing import Union, Literal

from sahi.utils.coco import Coco, CocoAnnotation, CocoPrediction

from label_studio_sdk import Prediction
import label_studio_sdk.converter.imports.coco

from labelstudio_tools import LabelStudioPlus
from patch_sahi import CocoPlus
from proj_utils import coco_poly_to_ls_rle, sort_prediction_result_SCANLINE

lsp = LabelStudioPlus.from_config('configs/ls_config.ichthyolith_sahi.json')


def coco_anno_to_ls_result(labeltype: Union[Literal['rectanglelabels'],Literal['brushlabels']],
                           anno:list, category_name:str,
                           img_width, img_height,
                           from_name="label", to_name="image",
                           score=None, meta:dict = None, obj_id:str = None):
    if labeltype == 'rectanglelabels':
        x, y, w, h = anno
        x = 100 * x / img_width
        w = 100 * w / img_width
        y = 100 * y / img_height
        h = 100 * h / img_height
        ls_result = dict(
            value=dict(rectanglelabels=[category_name],
                       x=x, y=y, width=w, height=h),
            from_name=from_name, to_name=to_name, type="rectanglelabels",
            original_width=img_width, original_height=img_height
        )
    elif labeltype == 'brushlabels':
        ls_rle = coco_poly_to_ls_rle(anno[0], img_width, img_height)
        ls_result = dict(
            type = "brushlabels",
            brushlabels = [category_name],
            value = dict(rle=ls_rle),
            format = "rle",
            original_width=img_width, original_height=img_height
        )
    else:
        raise ValueError(f'labeltype "{labeltype}" not recognized')

    if obj_id:
        ls_result['id'] = obj_id
    if score:
        ls_result['score'] = score
    if meta:
        ls_result['meta'] = meta

    return ls_result



def upload_fullframe_predictions(coco_path_or_dict, project='Fullframe Imagery'):
    fuecoco = CocoPlus.from_coco_dict_or_path(coco_path_or_dict)
    coco_image_by_filenamestem = {os.path.splitext(os.path.basename(img.file_name))[0]: img for img in fuecoco.images}
    lsp.set_project(project)
    fullframe_tasks = lsp.get_tasks(limit_fields_to=['id', 'data', 'predictions', 'annotations'])
    tasks_by_fullframeid = lsp.tasks_by_pk(fullframe_tasks,'fullframe_id')

    for fullframe_id,coco_img in coco_image_by_filenamestem.items():
        # TODO idempotent based on prediction names
        if fullframe_id not in tasks_by_fullframeid:
            print(f'{fullframe_id}... skipped')
            continue
        else:
            task = tasks_by_fullframeid[fullframe_id]

        print(f'{fullframe_id}: annots: {len(coco_img.annotations)}, preds: {len(coco_img.predictions)}')

        ls_annots = []
        coco_img.annotations = sort_prediction_result_SCANLINE(coco_img.annotations)
        for idx, anno in enumerate(coco_img.annotations):
            # ls_anno = coco_anno_to_ls_result(
            #     'rectanglelabels',
            #     anno.bbox, anno.category_name,
            #     obj_id=f'obj{idx+1:03}',
            #     img_width=coco_img.width, img_height=coco_img.height,
            #     from_name='label', to_name='image',
            #     meta = dict(obj_id=idx+1))
            ls_anno = coco_anno_to_ls_result(
                'brushlabels',
                anno.segmentation, anno.category_name,
                obj_id=f'obj{idx+1:03}',
                img_width=coco_img.width, img_height=coco_img.height,
                from_name='brush', to_name='image',
                #meta = dict(obj_id=idx+1),
                )
            ls_annots.append(ls_anno)

        ls_preds = []
        avg_score = []
        if fuecoco.name in [prediction['model_version'] for prediction in task['predictions']]:
            print(f'{fullframe_id} {fuecoco.name} ... skipped. Already Exists')
        else:
            coco_img.predictions = sort_prediction_result_SCANLINE(coco_img.predictions)
            for idx, pred in enumerate(coco_img.predictions):
                avg_score.append(pred.score)
                ls_pred = coco_anno_to_ls_result(
                    'rectanglelabels',
                    pred.bbox, pred.category_name,
                    obj_id = f'obj{idx+1:03}',
                    img_width=coco_img.width, img_height=coco_img.height,
                    from_name='label', to_name='image',
                    score=pred.score, meta=dict(obj_id=idx+1))
                ls_preds.append(ls_pred)

        if ls_annots:
            resp_anno = lsp.client.annotations.create(id=task['id'], result=ls_annots)

        if ls_preds:
            avg_score = sum(avg_score) / len(avg_score)
            resp_anno = lsp.client.predictions.create(task=task['id'], result=ls_preds, model_version=fuecoco.name, score=avg_score)
        return

def get_slice_param_key(slice_shape, slice_overlap):
    if isinstance(slice_shape, int):
        shape = f's{slice_shape}'
    else:
        shape = f's{slice_shape[0]}x{slice_shape[1]}'
    if isinstance(slice_overlap, float):
        overlap = f'o{int(100*slice_overlap)}'
    else:
        overlap = f'o{int(100*slice_overlap[0])}x{int(100*slice_overlap[1])}'
    return f'{shape}_{overlap}'



from patch_sahi import slice_coco


def slice_predictions_and_upload(
        lsp:LabelStudioPlus,
        coco_dict_or_path,
        sliced_image_outdir,
        sliced_coco_outfile,
        input_coco_imagedir = '',
        slice_shape=1024,
        slice_overlap=0.2,
        include_slicebox_pred = True,
        include_annotations = False,
        include_predictions = False,
        ):

    slice_param_str = get_slice_param_key(slice_shape, slice_overlap)

    if isinstance(slice_shape, int):
        slice_shape = (slice_shape,slice_shape)
    if isinstance(slice_overlap, float):
        slice_overlap = (slice_overlap,slice_overlap)

    sliced_image_outdir = sliced_image_outdir.format(SLICE_PARAM_STR=slice_param_str)

    out_ext = '.jpg'
    sliced_coco, original_size_mapper = slice_coco(
        coco_dict_or_path,
        image_dir=input_coco_imagedir,
        output_dir=sliced_image_outdir,
        slice_width=slice_shape[0],
        slice_height=slice_shape[1],
        overlap_width_ratio=slice_overlap[0],
        overlap_height_ratio=slice_overlap[1],
        min_area_ratio=0.05,
        out_ext=out_ext)

    sliced_coco_outfile = sliced_coco_outfile.format(SLICE_PARAM_STR=slice_param_str)
    with open(sliced_coco_outfile,'w') as f:
        json.dump(sliced_coco.json, f)

    image_by_sliceid = []
    for image in tqdm(sliced_coco.images):
        slice_id = Path(image.file_name).stem
        fullframe_id, slice_key_str = slice_id.rsplit('__',1)
        sample_id = fullframe_id.split('__')[0]
        slice_key = list(map(int,slice_key_str.split('_')))
        slice_offset = dict(x=slice_key[0], y=slice_key[1])
        slice_shape = dict(w=slice_key[2]-slice_key[0], h=slice_key[3]-slice_key[1])
        fullframe_shape = original_size_mapper[image.file_name]

        s3_key = f'fullframe_halfsizejpg__slices/{fullframe_id}/{slice_param_str}/{slice_key_str}{out_ext}'
        s3_url = lsp.s3key_to_url(s3_key)

        fullframe__s3url = lsp.s3key_to_url(f'fullframe_halfsizejpg/{fullframe_id}.jpg')
        fullframe_fullres__s3url = lsp.s3key_to_url(f'fullframe_fullres/{fullframe_id}.tif')

        task_data = dict(
            image = s3_url,
            slice_id = slice_id,
            shape = slice_shape,
            slice_offset = slice_offset,
            fullframe_id = fullframe_id,
            fullframe_shape = fullframe_shape,
            sample_id = sample_id,
            fullframe__s3url = fullframe__s3url,
        )
        if lsp.s3url_exists(fullframe__s3url):
            task_data['fullframe__s3url'] = fullframe__s3url
        if lsp.s3url_exists(fullframe_fullres__s3url):
            task_data['fullframe_fullres__s3url'] = fullframe_fullres__s3url

        filepath = os.path.join(sliced_image_outdir, image.file_name)

        s3obj_created = lsp.upload_s3url(filepath, s3_url, clobber=False)
        ls_id, task_exists, task_created = lsp.create_task(task_data, 'slice_id', use_cache=True)
        # WOOPS! UPDATE!
        # if task_exists and task_created is False:
        #     task = lsp.get_tasks(ids=[ls_id])[0]
        #     lsp.update_task(task, patch_data=dict(slice_offset=slice_offset))

        if task_created is True and include_slicebox_pred:
            # add slicebox
            x = 100 * slice_offset['x'] / fullframe_shape['w']
            w = 100 * slice_shape['w'] / fullframe_shape['w']
            y = 100 * slice_offset['y'] / fullframe_shape['h']
            h = 100 * slice_shape['h'] / fullframe_shape['h']
            ls_slicebox_pred = dict(value=dict(x=x, y=y, width=w, height=h),
                            readonly=True,
                           from_name="fullframe_slicebox", to_name="fullframe", type="rectangle",
                           original_width=fullframe_shape['w'], original_height=fullframe_shape['h'])
            lsp.client.predictions.create(
                        task=ls_id, result=[ls_slicebox_pred],
                        model_version='slicebox', score=0)

        if include_predictions:
            raise NotImplemented
            ls_annots = []
            avg_score = []
            for annot in image.predictions:
                avg_score.append(annot.score or 0)
                if annot.bbox:
                    ls_bbox_result = coco_anno_to_ls_result(
                        'rectanglelabels',
                        annot.bbox, annot.category_name,
                        image.width, image.height,
                        score=annot.score or 0,
                        from_name='tag2'  # TODO easier usage
                    )
                    ls_annots.append(ls_bbox_result)
                if annot.segmentation:
                    ls_rle_result = coco_anno_to_ls_result(
                        'brushlabels',
                        annot.segmentation, annot.category_name,
                        image.width, image.height,
                        score=annot.score or 0,
                        from_name='tag'  # TODO easier usage
                    )
                    ls_annots.append(ls_rle_result)

            if ls_annots:
                avg_score = sum(avg_score) / len(avg_score)
                resp_anno = lsp.client.predictions.create(
                    task=ls_id, result=ls_annots, score=avg_score,
                    model_version='train_v1', # TODO Model name
                )

        if include_annotations:
            raise NotImplemented
            ls_annots = []
            for annot in image.annotations:
                avg_score.append(annot.score or 0)
                if annot.bbox:
                    ls_bbox_result = coco_anno_to_ls_result(
                        'rectanglelabels',
                        annot.bbox, annot.category_name,
                        image.width, image.height,
                        from_name='tag2',  # TODO easier usage
                        meta = None  # TODO explore this to keep track of canonical IDs
                    )
                    ls_annots.append(ls_bbox_result)
                if annot.segmentation:
                    ls_rle_result = coco_anno_to_ls_result(
                        'brushlabels',
                        annot.segmentation, annot.category_name,
                        image.width, image.height,
                        from_name='tag',  # TODO easier usage
                        meta=None  # TODO explore this to keep track of canonical IDs
                    )
                    ls_annots.append(ls_rle_result)

            if ls_annots:
                resp_anno = lsp.client.annotations.create(
                    task=ls_id, result=ls_annots
                )


# lsp.set_project('Fullslide Seg Training v1')
# lsp.task_pk_datafields = 'slice_id'
# slice_predictions_and_upload(
#         lsp,
#         coco_dict_or_path='runs/inference_sahi/detect3/predictions.coco.json',
#         sliced_image_outdir = 'datasets/intermediary/sliced__{SLICE_PARAM_STR}',
#         sliced_coco_outfile = 'datasets/annotations/121N1of2__sliced_{SLICE_PARAM_STR}.coco.json',
#         input_coco_imagedir = 'datasets/intermediary/fullframe_halfsize_jpg',
#         slice_shape=1024, slice_overlap=0.2,
# )


# upload_fullframe_predictions('runs/inference_sahi/detect4/predictions.coco.json')
# upload_fullframe_predictions('runs/inference_sahi/detect5/predictions.coco.json')
# upload_fullframe_predictions('runs/inference_sahi/detect6/predictions.coco.json')
# upload_fullframe_predictions('runs/inference_sahi/detect7/predictions.coco.json')

upload_fullframe_predictions('datasets/annotations/fuecoco_FIXED4.json', project='Fullframe Mask Annotate')





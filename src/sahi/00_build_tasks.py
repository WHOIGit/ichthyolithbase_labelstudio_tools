import os
import io

from tqdm import tqdm
from natsort import natsorted
from PIL import Image
Image.MAX_IMAGE_PIXELS = 400_000_000  # 20k * 20k pixels

from core import LabelStudioPlus
lsp = LabelStudioPlus.from_config('configs/ls_config.ichthyolith_sahi.json')


S3_FULLFRAME_PATTERN = 'fullframe_halfsizejpg/{fullframe_id}{ext}'
S3_FULLFRAME_FULLRES_PATTERN = 'fullframe_fullres/{fullframe_id}{ext}'
S3_FULLFRAME_FULLRES_3D_PATTERN = 'fullframe_fullres_csv3d/{fullframe_id}{ext}'
S3_SLICE_PATTERN = 'slices/{fullframe_id}/{slice_str}{ext}'
S3_ROI_PATTERN = 'rois/{fullframe_id}/{roi_str}{ext}'



def fullframe_original_filename_to_fullframeID_ext(filename):
    filename = os.path.basename(filename)
    filename,ext = os.path.splitext(filename)
    filename = filename.replace('_picked_','_')  # ODP
    sample_id,image_id = filename.split('cm_N',1)
    sample_id = sample_id+'cm'
    image_id = 'N'+image_id
    image_id = image_id.replace('z','Z').replace('x','')
    image_id = image_id.replace('_2d','').replace('_3d','')
    fullframeID = f"{sample_id}__{image_id}"
    return fullframeID,ext


def downcast_fullframe_to_jpg(fullframe_fullres, save_as=None, quality=90, downscale_factor=2, method=Image.Resampling.LANCZOS):
    if isinstance(fullframe_fullres, str):
        img = Image.open(fullframe_fullres)
    else:
        img = fullframe_fullres.seek(0)

    new_shape = width, height = img.size
    if isinstance(downscale_factor, tuple) and len(downscale_factor)==2:
        new_shape = downscale_factor
    elif isinstance(downscale_factor, dict):
        new_shape = downscale_factor['w'],downscale_factor['h']
    elif isinstance(downscale_factor, int) or isinstance(downscale_factor,float):
        new_shape = (width // downscale_factor, height // downscale_factor)

    if new_shape != (width,height):
        img = img.resize(new_shape, method)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    jpg_img = Image.open(buffer)
    if save_as:
        buffer.seek(0)
        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        with open(save_as, 'wb') as f:
            f.write(buffer.read())
    return jpg_img


def task_paradata_to_task_data(lsp, task_paradata, s3upload=True, clobber=False):
    task_data = task_paradata.copy()

    def taskdata_modify(file_s3key_name, s3url_name):
        if file_s3key_name in task_data and task_data[file_s3key_name]:
            localfile, s3key = task_data.pop(file_s3key_name)
            s3url = lsp.s3key_to_url(s3key)
            task_data[s3url_name] = s3url
            if s3upload:
                lsp.upload_s3url(localfile, s3url, clobber=clobber)

    taskdata_modify('image__file_s3key', 'image')
    taskdata_modify('fullframe__file_s3key', 'fullframe__s3url')
    taskdata_modify('fullframe_fullres__file_s3key', 'fullframe_fullres__s3url')
    taskdata_modify('fullframe_fullres_3d__file_s3key', 'fullframe_fullres_3d__s3url')

    return task_data


def construct_fullframe_task_paradata(fullframe_fullres_local, fullframe_fullres_3d_local=None, downscale_factor=2):

    fullframe_id,ext = fullframe_original_filename_to_fullframeID_ext(fullframe_fullres_local)
    sample_id = fullframe_id.split('__',1)[0]

    fullres_filename = os.path.basename(fullframe_fullres_local)
    fullres_PIL = Image.open(fullframe_fullres_local)
    fullres_shape = fullres_PIL.size  # (width, height)
    fullres_shape = {'w': fullres_shape[0], 'h': fullres_shape[1]}

    fullframe_intermediary_dir = 'datasets/intermediary/fullframe_halfsize_jpg'
    fullframe_local = os.path.join(fullframe_intermediary_dir, f'{fullframe_id}.jpg')
    fullframe_PIL = downcast_fullframe_to_jpg(fullframe_fullres_local, save_as=fullframe_local, downscale_factor=downscale_factor)
    fullframe_shape = fullframe_PIL.size  # (width, height)
    fullframe_shape = {'w': fullframe_shape[0], 'h': fullframe_shape[1]}

    fullframe_fullres__file_s3key = fullframe_fullres_local, S3_FULLFRAME_FULLRES_PATTERN.format(fullframe_id=fullframe_id, ext=ext)
    fullframe__file_s3key = fullframe_local, S3_FULLFRAME_PATTERN.format(fullframe_id=fullframe_id, ext='.jpg')

    paradata =  dict(
        fullframe_id = fullframe_id,
        sample_id = sample_id,
        shape = fullframe_shape,
        fullframe_fullres_shape = fullres_shape,
        fullframe_fullres_filename=fullres_filename,

        # these contain both local path and s3key
        image__file_s3key = fullframe__file_s3key,
        fullframe_fullres__file_s3key = fullframe_fullres__file_s3key,
    )
    if fullframe_fullres_3d_local:
        assert os.path.isfile(fullframe_fullres_3d_local)
        paradata['fullframe_fullres_3d__file_s3key'] = fullframe_fullres_3d_local, S3_FULLFRAME_FULLRES_3D_PATTERN.format(fullframe_id=fullframe_id, ext='.csv')
    else:
        paradata['fullframe_fullres_3d__file_s3key'] = None

    return paradata


def create_fullframe_tasks(lsp:LabelStudioPlus, project, tiff_images, dry_run=False):
    lsp.set_project(project)
    lsp.task_pk_datafields = 'fullframe_id'
    lsp.cache_task_by_pk()
    to_be_created = []
    extant_tasks = []
    for tiff_image in tqdm(tiff_images):

        heightmap_csv = None
        if isinstance(tiff_image,tuple):
            tiff_image, heightmap_csv = tiff_image

        fullframe_id, ext = fullframe_original_filename_to_fullframeID_ext(tiff_image)
        fetched_task = lsp.task_exists(dict(fullframe_id=fullframe_id), 'fullframe_id', use_cache=True)
        if fetched_task:
            extant_tasks.append(fetched_task)
            continue

        fullframe_paradata = construct_fullframe_task_paradata(tiff_image, heightmap_csv)
        fullframe_taskdata = task_paradata_to_task_data(lsp, fullframe_paradata, s3upload=True, clobber=False)
        to_be_created.append(fullframe_taskdata)

    created_tasks = lsp.create_tasks(to_be_created, pk_datafields='fullframe_id', dry_run=dry_run) if to_be_created else []
    # todo print statements
    return created_tasks, extant_tasks


inputdata_dir = './datasets/input_data/'
input_tiff_and_csv = {}
for f in natsorted(os.listdir(inputdata_dir)):
    filename,ext = os.path.splitext(f)
    filename = filename.replace('_2d','').replace('_3d','')
    if filename in input_tiff_and_csv:
        input_tiff_and_csv[filename][ext] = os.path.join(inputdata_dir,f)
    else:
        input_tiff_and_csv[filename] = {ext:os.path.join(inputdata_dir,f)}
print(input_tiff_and_csv)
tiff_images = []
for tiff_csv in input_tiff_and_csv.values():
    tiff_images.append( (tiff_csv['.tif'],tiff_csv['.csv']) )
print(tiff_images)

#created_tasks, extant_tasks = create_fullframe_tasks(lsp, 'Fullframe Imagery', tiff_images, dry_run=False)
created_tasks, extant_tasks = create_fullframe_tasks(lsp, 'Fullframe Mask Annotate', tiff_images, dry_run=False)
print(f'Created {len(created_tasks)} new fullframe tasks, {len(extant_tasks)} already existed.')


def construct_slice_task_paradata(lsp, slice_file):
    # slice_id
    # sample_id + fullframe_id
    # image
    # shape {w=..., h=...}
    # offset {x=..., y=...}
    # fullframe_s3url
    # fullframe_shape {w=..., h=...}
    # fullframe_fullres_filename
    # fullframe_fullres_s3url
    # fullframe_fullres_shape {w=..., h=...}
    ...

def construct_roi_task_data(roi_file):
    # object_id
    # sample_id + fullframe_id
    # image
    # shape {w=..., h=...}
    # centerpoint {x=..., y=...}
    # fullframe_s3url
    # fullframe_shape {w=..., h=...}
    # fullframe_fullres_filename
    # fullframe_fullres_s3url
    # fullframe_fullres_shape {w=..., h=...}
    ...



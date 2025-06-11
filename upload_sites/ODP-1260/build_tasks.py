import os
from tqdm import tqdm

import upload_sites.common


def extract_fullslide_pid_metadata(path, add_description=True):
    import re
    # rois: {ROOT}/ODP-1260/1260B_101_17R_7W_74-75cm_g38um_N1of1_200x/unlabeled/1260B_101_17R_7W_74-75cm_g38um_N1of1_200x_obj00001_plane000.tif
    # full: {ROOT}/ODP-1260/1260B_101_17R_7W_74-75cm_g38um_N1of1_200x/ODP 1260_boxes_th=0.1300_size=0050u-2000u_1260B_101_17R_7W_74-75cm_g38um_N1of1_200x.tif
    metadata = {}
    is_roi = bool('_obj' in path)
    if is_roi:
        metadata_str = path.split('/')[-3]
        obj = path.split('_')[-2]  # obj00001
        obj = int(obj[3:])
    else:
        obj = None
        metadata_str = path.split('/')[-2]
        fullslide_split = path.split('_')
        fullslide_boxesth = fullslide_split[-10]  # th=0.1300
        fullslide_size = fullslide_split[-9]    # size=0050u-2000u
        boxesth = fullslide_boxesth.split('=')[1]
        size = fullslide_size.split('=')[1]
        metadata['fullslide__boxes_th'] = boxesth
        metadata['fullslide__size'] = size

    # metadata_str example: "1260B_101_17R_7W_74-75cm_g38um_N1of1_200x"
    metadata_str = 'ODP-'+metadata_str
    pattern = '(?P<site>ODP-\d+)(?P<hole>[A-Z]+)_(?P<internal_id>\d+)_(?P<core>[A-Za-z0-9]+)_(?P<section>[A-Za-z0-9]+)_(?P<interval>\d+-\d+cm)_(?P<fractioning>[A-Za-z0-9]+)_(?P<fullslideN>N\d+of\d+)_(?P<magnification>[A-Za-z0-9]+)'

    match = re.match(pattern, metadata_str)
    assert bool(match), f'metadata_str "{metadata_str}" did not match pattern "{pattern}". {match}'
    ext = match.groupdict()  # extracted
    metadata.update(ext)

    sample_id = '{site}_{hole}_{core}_{section}_{interval}'.format(**metadata)
    metadata['sample_id'] = sample_id
    fullslide_id = '{sample_id}_{fullslideN}'.format(**metadata)
    metadata['fullslide_id'] = fullslide_id
    if is_roi:
        metadata['object_id'] = '{fullslideN}_obj{object:05}'.format(object=obj, **metadata)
        pid = '{sample_id}_{object_id}'.format(**metadata)
        metadata['pid'] = pid
    else:
        pid = None

    if is_roi and add_description:
        s = ('internal_id={internal_id}, magnification={magnification}, size_fractioning={fractioning}\n'
             'Sample: {sample_id}\n'
             'Object: {object_id}'
             )
        metadata['description'] = s.format(**metadata)

    return fullslide_id, pid, metadata


def build_tasks(roi_listfile='ODP-1260_tifflist.txt',
                fullslide_listfile='ODP-1260_biglist.txt',
                ROOT='/user/esibert/ichthyolithBase/',
                bucket_name='ichthyolith'):
    with open(roi_listfile) as f:
        tiff_files = f.read().splitlines()
    with open(fullslide_listfile) as f:
        fullslide_files = f.read().splitlines()

    # list image files. sanstext are raw imagery, withtext has metadata-text and borders baked on
    withtext = [tiff for tiff in tiff_files if '/labeled/' in tiff]
    sanstext = [tiff for tiff in tiff_files if '/unlabeled/' in tiff]

    check1=set(tiff_files)-set(withtext)-set(sanstext)  # check all accouted for
    check2a, check2b = len(withtext), len({os.path.basename(tiff) for tiff in withtext})
    check3a, check3b = len(sanstext), len({os.path.basename(tiff) for tiff in sanstext})
    assert not check1
    assert check2a == check2b
    assert check3a == check3b

    tasks = upload_sites.common.build_tasks(sanstext, withtext, fullslide_files,
                                    extract_fullslide_pid_metadata, bucket_name, ROOT)
    return tasks

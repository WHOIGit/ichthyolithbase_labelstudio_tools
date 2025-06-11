import os
from tqdm import tqdm

import upload_sites.common

def extract_fullslide_pid_metadata(path, add_description=True):
    import re
    # rois: {ROOT}/DSDP-596/DSDP-596-P001-L01-1H-2W-5-7cm-g106_Hwell_N1of1_Mcompound_Oflat_I1_TzEDF-0_X5/final/focused/DSDP-596-P001-L01-1H-2W-5-7cm-g106_obj00001_edf.tif
    # full: {ROOT}/DSDP-596/DSDP-596-P001-L01-1H-2W-5-7cm-g106_Hwell_N1of1_Mcompound_Oflat_I1_TzEDF-0_X5/DSDP-596-P001-L01-1H-2W-5-7cm-g106_boxes_th=0.1800_size=0100u-4500u.jpg
    metadata = {}
    is_roi = bool('_obj' in path)
    if is_roi:
        metadata_str = path.split('/')[-4]
        obj = path.split('_')[-2]  # obj00001
        obj = int(obj[3:])
    else:
        obj = None
        metadata_str = path.split('/')[-2]
        fullslide_boxesth = path.split('_')[-2]  # th=0.1800
        fullslide_size = path.split('_')[-1]    # size=0100u-4500u.jpg
        boxesth = fullslide_boxesth.split('=')[1]
        size = fullslide_size.split('=')[1].split('.')[0]
        metadata['fullslide__boxes_th'] = boxesth
        metadata['fullslide__size'] = size

    # metadata_str example: "DSDP-596-P001-L01-1H-2W-5-7cm-g106_Hwell_N1of1_Mcompound_Oflat_I1_TzEDF-0_X5"
    pattern = '(?P<site>DSDP-\d+)-(?P<slide>P\d+)-(?P<internal_id>[A-Za-z0-9]+)-(?P<core>\d+H)-(?P<section>\d+W)-(?P<interval>\d+-\d+cm)-(?P<fractioning>g\d+)_Hwell_(?P<fullslideN>N\d+of\d+)_Mcompound_Oflat_I1_TzEDF-0_(?P<magnification>[A-Za-z0-9]+)'
    match = re.match(pattern, metadata_str)
    assert bool(match), f'metadata_str "{metadata_str}" did not match pattern "{pattern}". {match}'
    ext = match.groupdict()  # extracted
    metadata.update(ext)

    metadata['sample_id'] = '{site}_{slide}_{core}_{section}_{interval}'.format(**metadata)
    metadata['fullslide_id'] = fullslide_id = '{sample_id}_{fullslideN}'.format(**metadata)
    if is_roi:
        metadata['object_id'] = '{fullslideN}_obj{object:05}'.format(object=obj, **metadata)
        pid = '{sample_id}_{object_id}'.format(**metadata)
        metadata['pid'] = pid
    else:
        pid = None

    if is_roi and add_description:
        s = ('Sample: {sample_id}\n'
             'Object: {object_id}'
             'internal_id={internal_id}, magnification={magnification}, size_fractioning={fractioning}'
             )
        metadata['description'] = s.format(**metadata)

    return fullslide_id, pid, metadata


def build_tasks(roi_listfile='DSDP-596_tifflist.txt',
                fullslide_listfile='DSDP-596_biglist.txt',
                ROOT='/user/esibert/ichthyolithBase/',
                bucket_name='ichthyolith'):
    with open(roi_listfile) as f:
        tiff_files = f.read().splitlines()
    with open(fullslide_listfile) as f:
        fullslide_files = f.read().splitlines()

    withtext = [tiff for tiff in tiff_files if '/final/focused/' in tiff]
    sanstext = [tiff for tiff in tiff_files if tiff not in withtext]
    #print(tiff_files[:2])
    #print(fullslide_files[:2])

    print(check1:=set(tiff_files)-set(withtext)-set(sanstext))  # check all accouted for
    #print(check2a:=len(withtext), check2b:=len({os.path.basename(tiff) for tiff in withtext}))
    #print(check3a:=len(sanstext), check3b:=len({os.path.basename(tiff) for tiff in sanstext}))
    assert not check1
    #assert check2a == check2b
    #assert check3a == check3b
    tasks = upload_sites.common.build_tasks(sanstext=[], withtext=withtext,
                            metadata_extractor=extract_fullslide_pid_metadata,
                            fullslide_files=fullslide_files,
                            bucket_name=bucket_name,
                            ROOT=ROOT)
    return tasks

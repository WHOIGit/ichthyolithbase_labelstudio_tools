import os
from tqdm import tqdm


def extract_fullslide_pid_metadata(path):
    ...
    # TODO
    return fullslide_id, pid, metadata



def build_tasks():
    with open('DSDP-596_tifflist.txt') as f:
        tiff_files = f.read().splitlines()
    with open('DSDP-596_biglist.txt') as f:
        fullslide_files = f.read().splitlines()

    labeled = [tiff for tiff in tiff_files if '/final/focused/' in tiff]
    unlabeled = [tiff for tiff in tiff_files if '/unlabeled/' in tiff] # TODO
    #print(tiff_files[:2])
    #print(fullslide_files[:2])

    print(check1:=set(tiff_files)-set(labeled)-set(unlabeled))  # check all accouted for
    print(check2a:=len(labeled), check2b:=len({os.path.basename(tiff) for tiff in labeled}))
    print(check3a:=len(unlabeled), check3b:=len({os.path.basename(tiff) for tiff in unlabeled}))
    assert not check1
    assert check2a == check2b
    assert check3a == check3b

    ROOT = '/user/esibert/ichthyolithBase/'

    fullslide_polaroid_mapper = {}
    for fullslide_polaroid_file in fullslide_files:
        fullslide_dir = fullslide_polaroid_file.split('_')[-2]
        map_overview = dict(
            fullpath = ROOT+fullslide_polaroid_file,
            tiff2jpg = True,
            s3_key = 'fullslide_polaroid_jpg/{fullslide_id}.jpg'
        )
        fullslide_polaroid_mapper[fullslide_dir] = map_overview

    tasks = dict()
    for tiff_file in unlabeled:
        metadata, pid, fullslide_id = extract_fullslide_pid_metadata(tiff_file)
        fullslide_dir = tiff_file.split('_')[-4]  # TODO check this is -4
        task = dict(
            data = dict(
                pid = pid,
                fullslide_id = fullslide_id,
                **metadata,
            ),
            map_rois_jpg = dict(
                fullpath = ROOT+tiff_file,
                tiff2jpg = True,
                s3_key = f'rois_jpg/{pid}.jpg',
            ),
            map_rois_tiff = dict(
                fullpath = ROOT+tiff_file,
                tiff2jpg = False,
                s3_key = f'rois_tiff/{pid}.tiff',
            ),
        )
        if fullslide_dir in fullslide_polaroid_mapper.keys():
            task['map_fullslide_polaroid_jpg'] = fullslide_polaroid_mapper[fullslide_dir]
        tasks[pid] = task

    for tiff_file in labeled:
        _, pid, fullslide_id = extract_fullslide_pid_metadata(tiff_file)
        assert fullslide_id == tasks[pid]['data']['fullslide_id']
        assert fullslide_id in pid
        tasks[pid]['map_rois_polaroid_jpg'] = dict(
            fullpath = ROOT+tiff_file,
            tiff2jpg = True,
            s3_key = f'rois_polaroid_jpg/{pid}.jpg',
        )

    for task in tqdm(tasks.values()):
        add_s3_image_references(task)

    return tasks


def compute_task_metadata(path):
    import re
    # DSDP-596/DSDP-596-P001-L01-1H-2W-5-7cm-g106_Hwell_N1of1_Mcompound_Oflat_I1_TzEDF-0_X5/final/focused/DSDP-596-P001-L01-1H-2W-5-7cm-g106_obj00001_edf.tif
    # TODO from filename just need obj0000, all other metadata accessible from fullslide_dir i think
    fullslide_dir = path.split('/')[-4]
    # DSDP-596-P001-L01-1H-2W-5-7cm-g106_Hwell_N1of1_Mcompound_Oflat_I1_TzEDF-0_X5
    pattern = '(?P<site>DSDP-\d+)-(?P<slide>P\d+)-(?P<sediment>[A-Za-z0-9]+)-(?P<core>\d+H)-(?P<section>\d+W)-(?P<interval>\d+-\d+cm)-(?P<fraction>g\d+)_Hwell_N(?P<fullslideN>\d+)of(?P<fullslidesM>\d+)_Mcompound_Oflat_I1_TzEDF-0_(?P<magnification>[A-Za-z0-9]+)'
    match = re.match(pattern, fullslide_dir)
    assert bool(match), f'fullslide_dir "{fullslide_dir}" did not match pattern "{pattern}". {match}'
    metadata = dict(match.groupdict())  # extracted
    metadata['iodp'] = f"{metadata['core']}_{metadata['section']}_{metadata['interval']}"
    metadata['object'] = str(int(path.split('_')[-2][3:])) # todo check
    return metadata


def add_s3_image_references(task, bucket='ichthyolith'):
    s3_url = 's3://{bucket}/{key}'

    if 'map_rois_jpg' in task:
        key = task['map_rois_jpg']['s3_key']
        task['data']['image'] = s3_url.format(bucket=bucket, key=key)

    if 'map_rois_polaroid_jpg' in task:
        key = task['map_rois_polaroid_jpg']['s3_key']
        task['data']['image_polaroid'] = s3_url.format(bucket=bucket, key=key)

    if 'map_fullslide_polaroid_jpg' in task:
        key = task['map_fullslide_polaroid_jpg']['s3_key']
        task['data']['image_fullslide_polaroid'] = s3_url.format(bucket=bucket, key=key)




tasks = build_tasks()
print('done:', len(tasks))
from pprint import pprint
pprint(list(tasks.values())[0])

# TODO ask if pid can be simplified.
#      can we drop iodp (core section interval), magnification, fraction ?
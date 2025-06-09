import os
from tqdm import tqdm

def extract_fullslide_pid_metadata(path):
    ...
    # TODO
    return fullslide_id, pid, metadata

def extract_pid(path):
    pid = os.path.basename(path)   # removes directory name
    pid = os.path.splitext(pid)[0] # removes ".tif"
    pid = pid.rsplit('_',1)[0]     # removes "_plane000"
    pid = 'ODP-'+pid
    return pid

def extract_fullslide_id(path, idx):
    return 'ODP-'+path.split('/')[idx]

def build_tasks():
    with open('ODF-1260_tifflist.txt') as f:
        tiff_files = f.read().splitlines()
    with open('ODF-1260_biglist.txt') as f:
        fullslide_files = f.read().splitlines()

    labeled = [tiff for tiff in tiff_files if '/labeled/' in tiff]
    unlabeled = [tiff for tiff in tiff_files if '/unlabeled/' in tiff]
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
        fullslide_id = extract_fullslide_id(fullslide_polaroid_file,-2)
        map_overview = dict(
            fullpath = ROOT+fullslide_polaroid_file,
            tiff2jpg = True,
            s3_key = f'fullslide_polaroid_jpg/{fullslide_id}.jpg'
        )
        fullslide_polaroid_mapper[fullslide_id] = map_overview

    tasks = dict()
    for tiff_file in unlabeled:
        pid = extract_pid(tiff_file)
        fullslide_id = extract_fullslide_id(tiff_file,-3)
        task = dict(
            data = dict(
                pid = pid,
                fullslide_id = fullslide_id
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
        if fullslide_id in fullslide_polaroid_mapper.keys():
            task['map_fullslide_polaroid_jpg'] = fullslide_polaroid_mapper[fullslide_id]
        tasks[pid] = task

    for tiff_file in labeled:
        pid = extract_pid(tiff_file)
        fullslide_id = extract_fullslide_id(tiff_file,-3)
        assert fullslide_id == tasks[pid]['data']['fullslide_id']
        assert fullslide_id in pid
        tasks[pid]['map_rois_polaroid_jpg'] = dict(
            fullpath = ROOT+tiff_file,
            tiff2jpg = True,
            s3_key = f'rois_polaroid_jpg/{pid}.jpg',
        )

    for task in tqdm(tasks.values()):
        compute_task_metadata(task)
        add_s3_image_references(task)

    return tasks


def compute_task_metadata(task):
    import re
    # ODP-1260B_101_17R_7W_74-75cm_g38um_N1of1_200x_obj00001
    pattern = '(?P<site>ODP-[A-Za-z0-9]+)_(?P<sediment>[A-Za-z0-9]+)_(?P<core>[A-Za-z0-9]+)_(?P<section>[A-Za-z0-9]+)_(?P<interval>\d+-\d+cm)_(?P<fraction>[A-Za-z0-9]+)_N(?P<fullslideN>\d+)of(?P<fullslidesM>\d+)_(?P<magnification>[A-Za-z0-9]+)'
    fullslide_id = task['data']['fullslide_id']
    match = re.match(pattern, fullslide_id)
    assert bool(match), f'fullslide_id "{fullslide_id}" did not match pattern "{pattern}". {match}'
    ext = match.groupdict()  # extracted
    ext['hole'] = ext['site'][-1]  # the B of ODP-1260B
    ext['site'] = ext['site'][:-1] #

    task['data'].update(ext)
    task['data']['iodp'] = f"{ext['core']}_{ext['section']}_{ext['interval']}"
    task['data']['object'] = str(int(task['data']['pid'].split('_')[-1][3:]))

    if 'map_fullslide_polaroid_jpg' in task:
        # {ROOT}/ODP-1260/1260B_101_17R_7W_74-75cm_g38um_N1of1_200x/ODP 1260_boxes_th=0.1300_size=0050u-2000u_1260B_101_17R_7W_74-75cm_g38um_N1of1_200x.tif
        fullslide_path = task['map_fullslide_polaroid_jpg']['fullpath']
        fullslide_split = os.path.basename(fullslide_path).split('_')
        boxesth = fullslide_split[-10] # th=0.1300
        size = fullslide_split[-9]   # size=0050u-2000u
        boxesth = boxesth.split('=')[1]
        size = size.split('=')[1]
        task['data']['fullslide__boxes_th'] = boxesth
        task['data']['fullslide__size'] = size

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
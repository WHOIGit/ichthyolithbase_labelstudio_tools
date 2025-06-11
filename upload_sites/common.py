from tqdm import tqdm

# TODO fullslide_files withtext vs sanstext
# todo what if you only have withtext tiffs?
def build_tasks(sanstext:list[str],
                withtext: list[str],
                fullslide_files:list[str],
                metadata_extractor:callable,
                bucket_name:str,
                ROOT='/user/esibert/ichthyolithBase/') -> list[dict]:

    fullslide_withtext_mapper = {}
    for fullslide_withtext_file in fullslide_files:
        fullslide_id,_,_ = metadata_extractor(fullslide_withtext_file)
        map_overview = dict(
            filepath = ROOT+fullslide_withtext_file,
            tiff2jpg = True,
            s3_key = f'fullslide_withtext_jpg/{fullslide_id}.jpg'
        )
        fullslide_withtext_mapper[fullslide_id] = map_overview

    tasks = dict()

    for tiff_file in sanstext:
        fullslide_id, pid, metadata = metadata_extractor(tiff_file)
        task = dict(
            pid = pid,
            data = metadata,
            map_rois_jpg = dict(
                filepath = ROOT+tiff_file,
                tiff2jpg = True,
                s3_key = f'rois_jpg/{pid}.jpg',
            ),
            map_rois_tiff = dict(
                filepath = ROOT+tiff_file,
                tiff2jpg = False,
                s3_key = f'rois_tiff/{pid}.tiff',
            ),
        )
        if fullslide_id in fullslide_withtext_mapper.keys():
            task['map_fullslide_withtext_jpg'] = fullslide_withtext_mapper[fullslide_id]
        tasks[pid] = task

    for tiff_file in withtext:
        fullslide_id, pid, _ = metadata_extractor(tiff_file)
        assert fullslide_id == tasks[pid]['data']['fullslide_id']
        assert fullslide_id in pid
        tasks[pid]['map_rois_withtext_jpg'] = dict(
            filepath = ROOT+tiff_file,
            tiff2jpg = True,
            s3_key = f'rois_withtext_jpg/{pid}.jpg',
        )

    if bucket_name:
        for task in tasks.values():
            add_s3_image_references(task, bucket_name)

    tasks = sorted(tasks.values(), key=lambda t: t['pid'])

    return tasks


def add_s3_image_references(task, bucket_name:str='ichthyolith'):
    s3_url = 's3://{bucket}/{key}'

    if 'map_rois_jpg' in task:
        key = task['map_rois_jpg']['s3_key']
        task['map_rois_jpg']['bucket'] = bucket_name
        task['data']['image'] = s3_url.format(bucket=bucket_name, key=key)

    if 'map_rois_withtext_jpg' in task:
        key = task['map_rois_withtext_jpg']['s3_key']
        task['map_rois_withtext_jpg']['bucket'] = bucket_name
        task['data']['image_withtext'] = s3_url.format(bucket=bucket_name, key=key)

    if 'map_fullslide_withtext_jpg' in task:
        key = task['map_fullslide_withtext_jpg']['s3_key']
        task['map_fullslide_withtext_jpg']['bucket'] = bucket_name
        task['data']['image_fullslide_withtext'] = s3_url.format(bucket=bucket_name, key=key)



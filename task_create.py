#!/usr/bin/env python3

import argparse
from io import BytesIO
import importlib

from PIL import Image
from tqdm import tqdm
from label_studio_sdk import LabelStudio

import utils


def ls_create_task(ls_client, project: int, task:dict, check_exists=True, dry_run=False):
    pid = task['pid']
    ls_id, task_exists, task_created = None, False, False

    # check if task already exists
    if check_exists:
        if isinstance(check_exists, list):
            first = check_exists[0]
            if isinstance(first, str):
                if pid in check_exists:
                    task_exists = True
            elif isinstance(first, dict):
                for taskdict in check_exists:
                    if pid == taskdict['data']['pid']:
                        task_exists = True
                        ls_id = taskdict['id']
                        break
            return ls_id, task_exists, task_created

        elif extant_tasks:=utils.ls_get_filtered_tasks(ls_client, project, 'pid', pid):
            assert len(extant_tasks)==1, f'Error: Multiple tasks with pid "{pid}" exist for this project!'
            ls_id = extant_tasks[0]['id']
            task_exists = True
            return ls_id, task_exists, task_created

    if not dry_run:
        # todo fix: this sometimes errors out with httpx.RemoteProtocolError: Server disconnected without sending a response.
        new_task = ls_client.tasks.create(project=project, data=task)
        ls_id = new_task.id
    else:
        ls_id = None
    task_created = True

    return ls_id, task_exists, task_created


def tiff_to_s3(filepath:str, s3_client_config, s3_key:str, s3_bucket=None, as_jpg=True, clobber=False, dry_run=False) -> (bool,bool):
    """Convert TIFF image to JPEG and upload to S3."""
    if not (filepath.lower().endswith('.tif') or filepath.lower().endswith('.tiff')):
        raise ValueError(f"Input file must be TIFF format: {filepath}")

    if as_jpg and not s3_key.lower().endswith('.jpg'):
        raise ValueError(f"Output key must end with .jpg: {s3_key}")

    object_exists, object_created = False,False
    if s3_key.startswith('s3://'):
        assert s3_bucket is None
        s3_bucket, s3_key = utils.s3_url_to_bucket_and_key(s3_key)
    client, s3_bucket = utils.s3_client_and_bucket(s3_client_config, bucket=s3_bucket)

    # query for existing s3 object
    object_exists = utils.s3_object_exists(s3_client_config, s3_key, s3_bucket)

    if not clobber:   # skip uploads if file exists in s3 already
        if object_exists:
            return object_exists, object_created

    if not as_jpg:  # upload as-tiff
        if not dry_run:
            s3_bucket.upload_file(filepath, s3_key)
        object_created = True
        return object_exists, object_created

    # convert to jpg
    with Image.open(filepath) as tiff_image:
        jpeg_image = tiff_image.convert("RGB")

        # Save to BytesIO buffer
        byte_stream = BytesIO()
        jpeg_image.save(byte_stream, format='JPEG', quality=90)
        byte_stream.seek(0)

        # Upload to S3
        if not dry_run:
            client.Object(s3_bucket.name, s3_key).put(Body=byte_stream, ContentType='image/jpeg')
        object_created = True
    return object_exists, object_created


def main():
    parser = argparse.ArgumentParser(description="Upload ichthyolith image data to S3 and create annotation tasks")
    parser.add_argument('CONFIG', metavar='JSON', help="path to upload config file")
    parser.add_argument('--s3_config', help="Path to JSON file containing S3 configuration")
    parser.add_argument('--ls_config', help="Path to JSON file with Label Studio configuration")
    parser.add_argument('--dry-run', action='store_true', help="Don't actually upload anything, just give stats")
    parser.add_argument('--slice', nargs=2, metavar=('START','STOP'), default=('0','None'), help='operate on subset of tasks')
    # TODO upload predictions from csv
    args = parser.parse_args()

    config = utils.load_config(args.CONFIG)
    assert 'roi_listfile' in config
    assert 'fullslide_listfile' in config
    assert 'task_builder_module' in config
    if 'ROOT' not in config: config['ROOT'] = ''

    task_builder_module = importlib.import_module(config['task_builder_module'])

    tasks = task_builder_module.build_tasks(
        config['roi_listfile'],
        config['fullslide_listfile'],
        config['ROOT']
    )

    #if args.slice:
    start,stop = args.slice  # default is ("0","None")
    start = int(start) if str(start).isdigit() else 0
    stop = int(stop) if str(stop).isdigit() else None
    tasks = tasks[slice(start,stop)]

    if args.s3_config:
        s3_config = utils.load_config(args.s3_config)
        s3_client, bucket = utils.s3_client_and_bucket(s3_config, config['bucket'])

    if args.ls_config:
        ls_config = utils.load_config(args.ls_config)
        ls_project = ls_config['project']
        ls_client =  LabelStudio(base_url=ls_config['host'], api_key=ls_config['token'])

    # TODO summary struct, per-pid entry, also tracks raised errors
    report = dict(pids=[],
                  ls_extant_tasks=[], ls_created_tasks=[], ls_success=[], ls_ids=[],
                  s3_extant_objects=[], s3_created_objects=[], s3_success=[])
    for task in tqdm(tasks, initial=start, total=len(tasks)+start):
        pid = task['pid']
        report['pids'].append(pid)

        if args.s3_config:
            task_s3_extant_objects = []
            task_s3_created_objects = []
            task_s3_success = []
            for mapper_key in ['map_rois_jpg',
                               'map_rois_withtext_jpg',
                               'map_fullslide_withtext_jpg']:
                if mapper_key in task:
                    try:
                        filepath:str = task[mapper_key]['filepath']
                        as_jpg:bool = task[mapper_key]['tiff2jpg']
                        bucket:str = task[mapper_key]['bucket']
                        s3_key:str = task[mapper_key]['s3_key']
                        object_exists, object_created = tiff_to_s3(filepath,
                            s3_client, s3_key, bucket, as_jpg, clobber=False, dry_run=args.dry_run)

                        task_s3_extant_objects.append(object_exists)
                        task_s3_created_objects.append(object_created)
                        task_s3_success.append(True)
                    except Exception as e:
                        task_s3_extant_objects.append(None)
                        task_s3_created_objects.append(None)
                        task_s3_success.append(e)
                        print(type(e), e)
                        #raise e

            report['s3_extant_objects'].append(task_s3_extant_objects)
            report['s3_created_objects'].append(task_s3_created_objects)
            report['s3_success'].append(task_s3_success)

        if args.ls_config:
            #try:
                ls_id, task_exists, task_created = ls_create_task(
                        ls_client, ls_project, task['data'],
                        check_exists=True, dry_run=args.dry_run)
                report['ls_extant_tasks'].append(task_exists)
                report['ls_created_tasks'].append(task_created)
                report['ls_ids'].append(ls_id)
                report['ls_success'].append(True)
            # except Exception as e:
            #     report['ls_extant_tasks'].append(None)
            #     report['ls_created_tasks'].append(None)
            #     report['ls_ids'].append(None)
            #     report['ls_success'].append(e)
            #     print(type(e),e)
            #     #raise e

        #break # single task

    #print(report)

    if not args.ls_config:
        report.pop('ls_extant_tasks')
        report.pop('ls_created_tasks')
        report.pop('ls_ids')
        report.pop('ls_success')
    if not args.s3_config:
        report.pop('s3_extant_objects')
        report.pop('s3_created_objects')
        report.pop('s3_success')

    # TODO summarize task creation
    # todo output file with an entry per pid
    #import pandas as pd
    #df = pd.DataFrame(report)
    #print(df.T)



if __name__ == "__main__":
    main()
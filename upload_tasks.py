#!/usr/bin/env python3

import argparse
import json
import os
from io import BytesIO
from typing import Dict, List, Any

import boto3
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


# Metadata fields to include in tasks
METADATA_FIELDS = [
    'pid', 'site', 'Site', 'Depth', 'Sample #', 'Object #',
    'slide', 'sediment', 'core', 'section', 'interval', 'fraction',
    'iodp', 'obj', 'magnification', 'Slide Hole', 'Slide Hole Max',
    'Notes on Coding'
]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load AWS credentials from a JSON config file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def build_task(row: pd.Series, codegroups: List[str], s3_pattern: str, prediction: bool = True):
    """Create a task dictionary from a dataframe row."""

    # Create initial annotations from codegroups
    initial_annotations = []
    for codegroup in codegroups:
        if codegroup in row and not pd.isna(row[codegroup]):
            annotation = {
                'type': 'taxonomy',
                'to_name': 'image',
                'from_name': codegroup,
                'value': {'taxonomy': [[row[codegroup]]]}
            }
            initial_annotations.append(annotation)

    # Create data dictionary with image URL
    data = {'image': s3_pattern.format(row.pid)}

    # Add metadata fields
    for field in METADATA_FIELDS:
        if field in row and not pd.isna(row[field]):
            val = row[field]
            if val:  # Only add non-empty values
                data[field] = val

    # Create task structure
    task = {'data': data}

    # Add annotations or predictions
    if initial_annotations:
        task['annotations'] = [{'result': initial_annotations}]

    return task


def ls_create_task(host: str, token: str, project: int, task:dict):
    url = host + '/api/tasks'
    headers = {"Content-Type": "application/json", "Authorization": f"Token {token}"}
    payload = task.copy()

    # can't create tasks with annotations or predictions right off the bat
    if 'annotations' in payload: payload.pop('annotations')

    payload['project'] = project

    response = requests.post(url, json=payload, headers=headers)

    # Check if the request was successful
    if response.status_code != 201:
        print(f"Failed to retrieve data: {response.status_code} {response.json()}")
    return response.json()['id']


def ls_create_annotation(host: str, token: str, task: dict, task_id:int):
    url = host + '/api/tasks/{id}/annotations'
    url = url.format(id=task_id)
    headers = {"Content-Type": "application/json", "Authorization": f"Token {token}"}
    payload = task['annotations'][0]

    response = requests.post(url, json=payload, headers=headers)

    # Check if the request was successful
    if response.status_code != 201:
        print(f"Failed to retrieve data: {response.status_code} {response.json()}")

    return response.json()['id']


def ls_create_prediction(host: str, token: str, task: dict, task_id:int):
    url = host + '/api/predictions'
    headers = {"Content-Type": "application/json", "Authorization": f"Token {token}"}

    payload = dict(
        task = task_id,
        result = task['annotations'][0]['result'],
        score = 0.666,
        model_version = 'human'
    )

    response = requests.post(url, json=payload, headers=headers)

    # Check if the request was successful
    if response.status_code != 201:
        print(f"Failed to retrieve data: {response.status_code} {response.json()}")

    return response.json()['id']


def tiff_to_jpg_s3(input_path: str, output_key: str, bucket: str, s3_client) -> None:
    """Convert TIFF image to JPEG and upload to S3."""
    if not (input_path.lower().endswith('.tif') or input_path.lower().endswith('.tiff')):
        raise ValueError(f"Input file must be TIFF format: {input_path}")

    if not output_key.lower().endswith('.jpg'):
        raise ValueError(f"Output key must end with .jpg: {output_key}")

    with Image.open(input_path) as tiff_image:
        jpeg_image = tiff_image.convert("RGB")

        # Save to BytesIO buffer
        byte_stream = BytesIO()
        jpeg_image.save(byte_stream, format='JPEG', quality=90)
        byte_stream.seek(0)

        # Upload to S3
        s3_client.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=byte_stream,
            ContentType='image/jpeg'
        )


def upload_image(row, task, s3_client, bucket) -> None:
    """Upload a single image from the dataframe to S3."""
    pid = row.pid
    assert pid==task['data']['pid']

    if 'path' not in row:
        print(f"Warning: No 'path' field for pid {pid}")
        return

    input_path = row.path

    if not os.path.isfile(input_path):
        print(f"Warning: File not found: {input_path}")
        raise FileNotFoundError(input_path)

    s3_url = task['data']['image']
    output_key = s3_url.split('/',3)[-1]  # s3_url.replace('s3://amplify-scratch/', '')

    tiff_to_jpg_s3(input_path, output_key, bucket, s3_client)



def main():
    parser = argparse.ArgumentParser(description="Upload ichthyolith image data to S3 and create annotation tasks")
    parser.add_argument('CSV', help="Path to CSV file containing image data")
    parser.add_argument('--s3_config', help="Path to JSON file containing S3 configuration")
    parser.add_argument('--ls_config', help="Path to JSON file with Label Studio configuration")
    parser.add_argument('--denticle-codes', required=True, default='../denticle_codes.json', help="Path to denticle codes JSON file (default: ../denticle_codes.json)")
    parser.add_argument('--prediction', action='store_true', help="Create predictions instead of annotations")

    args = parser.parse_args()

    df = pd.read_csv(args.CSV)

    # Load configuration and data
    print("Loading configuration and data...")
    cats = load_config(args.denticle_codes)
    codegroups = list(cats.keys())

    # Create S3 client
    if args.s3_config:
        s3_config = load_config(args.s3_config)
        bucket = s3_config.pop('bucket')
        s3_client = boto3.client('s3', **s3_config)

    if args.ls_config:
        ls_config = load_config(args.ls_config)

    # Create tasks for all rows
    print("Creating tasks...")
    s3_pattern = 's3://amplify-scratch/paleofish/{}.jpg'
    for _, row in tqdm(df.iterrows()):
        task = build_task(row, codegroups, s3_pattern=s3_pattern, prediction=args.prediction)

        if args.ls_config:
            task_id = ls_create_task(task=task, **ls_config)
            project = ls_config.pop('project')
            if args.prediction:
                ls_create_prediction(**ls_config, task=task, task_id=task_id)
            else:
                ls_create_annotation(**ls_config, task=task, task_id=task_id)

        if args.s3_config:
            upload_image(row, task, s3_client, bucket)

        break


if __name__ == "__main__":
    main()
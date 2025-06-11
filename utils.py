import json
from typing import Dict, Any

from tqdm import tqdm
import boto3, botocore
import requests


def load_config(config_path: str) -> Dict[str, Any]:
    """Load AWS credentials from a JSON config file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def s3_url_to_bucket_and_key(url:str) -> (str,str):
    # eg: # s3://ichthyolith/rois_jpg/abcxyz.jpg
    assert url.startswith('s3://')
    bucket = url.split('/')[2]  # ichthyolith
    key = url.split('/',3)[-1]  # rois_jpg/abcxyz.jpg
    return bucket, key


def s3_client_and_bucket(client_config, bucket=None):
    """Instantiates an s3 connection. If already instantiated, nothing happens"""
    if isinstance(client_config, dict):
        config = client_config.copy()
        if bucket is None and 'bucket' in config:
            bucket= config.pop('bucket')
        elif 'bucket' in config:
            config.pop('bucket')
        client = boto3.resource('s3', **config)
    else:
        client = client_config  # boto3.resources.factory.s3.Bucket or botocore.client.S3

    if isinstance(bucket,str):
        bucket = client.Bucket(bucket)
    return client, bucket


def ls_get_filtered_tasks(ls_client, project, field, value, operator='equal', fieldtype='String', resolve_uri=False):
    host = ls_client._client_wrapper._base_url
    token = ls_client._client_wrapper._tokens_client._api_key
    url = host + '/api/tasks'
    headers = {"Content-Type": "application/json", "Authorization": f"Token {token}"}

    filter = {"conjunction": "and",
              "items": [{"filter": f"filter:tasks:data.{field}",
                         "operator": operator,
                         "value": value,
                         "type": fieldtype}]
              }
    query = dict(filters=filter)
    query = json.dumps(query)
    payload = dict(
        project=project,
        resolve_uri=resolve_uri,
        query=query
    )
    response = requests.get(url, params=payload, headers=headers)
    if response.status_code!=200:
        raise ValueError(f"Status Code {response.status_code}: {response.json()['detail']}")
    return response.json()['tasks']


def s3_list_objects(client_config, bucket=None, prefix=None):
    client, bucket = s3_client_and_bucket(client_config, bucket)

    object_keys = []
    if prefix:
        for obj in tqdm(bucket.objects.filter(Prefix=prefix)):
            object_keys.append(obj.key)
    else:
        for obj in tqdm(bucket.objects.all()):
            object_keys.append(obj.key)

    return object_keys


def s3_object_exists(client_config, key, bucket=None):
    if key.startswith('s3://'):     # s3://ichthyolith/rois_jpg/abcxyz.jpg
        bucket, key = s3_url_to_bucket_and_key(key)
    client, bucket = s3_client_and_bucket(client_config, bucket)
    obj = client.Object(bucket.name, key)
    try: obj.load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            raise e
    return True


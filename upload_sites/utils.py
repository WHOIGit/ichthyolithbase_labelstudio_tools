import json
from tqdm import tqdm
import boto3

s3_config_json = '../configs/s3_config.ichthyolith.json'
s3_config_json = '../configs/s3_config.amplify-scratch.json'

def load_config(config_path):
    """Load AWS credentials from a JSON config file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def s3_list_objects(aws_config, prefix=None, bucket_name=None):
    object_keys = []
    if bucket_name is None and 'bucket' in aws_config:
        bucket_name = aws_config.pop('bucket')
    elif 'bucket' in aws_config:
        aws_config.pop('bucket')

    s3 = boto3.resource('s3', **aws_config)
    bucket = s3.Bucket(bucket_name)
    if prefix:
        for obj in tqdm(bucket.objects.filter(Prefix=prefix)):
            object_keys.append(obj.key)
    else:
        for obj in tqdm(bucket.objects.all()):
            object_keys.append(obj.key)

    return object_keys

s3_config = load_config(s3_config_json)

objs = s3_list_objects(s3_config, prefix='paleofish')
print(objs[:5])
print(len(objs))
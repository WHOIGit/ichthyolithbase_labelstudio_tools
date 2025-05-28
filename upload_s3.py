#!/usr/bin/env python

import argparse
import boto3
import os
import json


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Upload a file to an S3 bucket.')

    parser.add_argument('FILE', help='Path to the file to upload')

    parser.add_argument('--bucket', help='S3 bucket name')
    parser.add_argument('--prefix', help='Optional key prefix (e.g., folder/) in the bucket')
    parser.add_argument('--key', help='Full S3 object key (overrides prefix + filename)')
    parser.add_argument('--config', required=True, help='Path to credentials config JSON file')

    return parser.parse_args()


def load_config(config_path):
    """Load AWS credentials from a JSON config file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def upload_file_to_s3(file_path, bucket, key, aws_config):
    """Upload a file to S3 using provided AWS config."""
    client = boto3.client('s3', **aws_config)
    client.upload_file(file_path, bucket, key)
 

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    assert os.path.isfile(args.FILE)
    
    config = load_config(args.config)
    
    if 'bucket' in config and args.bucket is None:
        args.bucket = config.pop('bucket')
    
    if args.key:
        s3_key = args.key
    else:
        filename = os.path.basename(args.FILE)
        prefix = args.prefix.rstrip('/') + '/' if args.prefix else ''
        s3_key = f"{prefix}{filename}"

    upload_file_to_s3(args.FILE, args.bucket, s3_key, config)


if __name__ == '__main__':
    main()
    

#!/usr/bin/env python

import os
import io
import json
import time
import argparse
import datetime as dt

import requests
from tqdm import tqdm
from label_studio_sdk.client import LabelStudio
from label_studio_sdk.data_manager import Filters, Column, Type, Operator


# values used to check if sub-views necessary
TIMEOUT_SECONDS=30
ITEMS_PER_SECOND=500



def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Create a snapshot of Label Studio project tasks.')
    parser.add_argument('--host', required=True, help='Label Studio URL (e.g., https://labelstudio.example.com)')
    parser.add_argument('--token', '-t', required=True, help='File with API key for Label Studio (available in Account & Settings > Access Tokens)')
    parser.add_argument('--project', '-p', metavar='ID', required=True, type=int, help='Project ID to snapshot')
    parser.add_argument('--fields', '-f', metavar='F', nargs='*', required=True, help='Annotation fields to cache. A line-delimited text file also accepted')
    #parser.add_argument('--views', type=int, nargs='*', help='Default is auto')
    return parser.parse_args()


def read_token(path):
    with open(path) as f:
        return f.read().strip()


def read_fields(field_args):
    fields = []
    for field in field_args:
        if os.path.isfile(field):
            with open(field) as f:
                fields.extend( f.read().strip().splitlines() )
        else:
            fields.append(field)
    return fields





def timeout_views_required(host:str, token:str, project_id:int, timeout_seconds=30, items_per_second=500):

    def project_counts(host:str, token:str, project_id:int):
        url = host + '/api/projects/counts'
        headers = {"Content-Type": "application/json", "Authorization":f"Token {token}"}  
        params = dict(ids=project_id)
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.json()['results'][0]
        else:
            print(response.status_code, response.json())

    counts = project_counts(host, token, project_id)
    timeout_views = counts['total_annotations_number'] // (items_per_second * timeout_seconds)
    print(f'Project has {counts} total annotations')
    return timeout_views+1



def update_cachelabel(host:str, token:str, project_id:int, control_tag:str, with_counters:bool=False, view_id:int=None, include:list=[], exclude:list=[]):
    url = host + '/api/dm/actions'  
    headers = {"Content-Type": "application/json", "Authorization":f"Token {token}"}  
    params = dict(id='cache_labels', project=project_id)
    payload = dict(
        source = 'Annotations',
        project = project_id,
        control_tag = control_tag,
        with_counters = 'Yes' if with_counters else 'No',
        )
    
    if view_id:
        params['tabID'] = view_id
    
    if include:
        payload['selectedItems'] = dict(all=False, included=include)
    else:
        payload['selectedItems'] = dict(all=True, excluded=exclude)
    
    response = requests.post(url, params=params, json=payload, headers=headers)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve data: {response.status_code} {response.json()}")


def update_cachelabels(host:str, token:str, project_id:int, control_tags:list, with_counters:bool=False, timeout_views='auto'):
    
    if timeout_views == 'auto':
        timeout_views = timeout_views_required(host, token, project_id, TIMEOUT_SECONDS, ITEMS_PER_SECOND)
        if timeout_views <=1:
            timeout_views = []
        else:
            # create N views over which tasks/annotations are distributed
            # will also need to clean these up afterwards, optionally
            raise NotImplementedError

    pbar = tqdm(control_tags)
    for tag in pbar:
        if timeout_views:
            for view_id in timeout_views:
                pbar.set_description(f'Updating "{tag}" (view_id={view_id})')
                update_cachelabel(LABEL_STUDIO_HOST, API_KEY, PROJECT_ID, tag, with_counters, view_id=view_id)
        else:
            pbar.set_description(f'Updating "{tag}"')
            update_cachelabel(host, token, project_id, tag, with_counters)


def main():
    # Parse command-line arguments
    args = parse_arguments()
    token = read_token(args.token)
    fields = read_fields(args.fields)
    
    # Connect to the Label Studio API
    ls = LabelStudio(base_url=args.host, api_key=token)

    # Get the project
    project = ls.projects.get(id=args.project)

    # Update the fields
    update_cachelabels(args.host, token, args.project, fields)
    

if __name__ == '__main__':
    main()
    

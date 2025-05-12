#!/usr/bin/env python

import os
import io
import json
import time
import argparse
import datetime as dt
import label_studio_sdk
from label_studio_sdk.client import LabelStudio
import pandas as pd
import httpx
from label_studio_sdk.data_manager import Filters, Column, Type, Operator



def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Create a snapshot of Label Studio project tasks.')
    parser.add_argument('--host', required=True, help='Label Studio URL (e.g., https://labelstudio.example.com)')
    parser.add_argument('--token',  required=True, help='File with API key for Label Studio (available in Account & Settings > Access Tokens)')
    parser.add_argument('--project', '-p', metavar='ID', required=True, type=int, help='Project ID to snapshot')
    parser.add_argument('--snapshot', metavar='ID', type=int, help='Download an existing snapshot')
    parser.add_argument('--filter', help='Json file defining task filtering')
    parser.add_argument('--nocleanup', action='store_true', help='Do not auto-remove Snapshot or filter View after download')
    parser.add_argument('--outdir', default='.', help='Output directory for downloaded task-data csv')
    parser.add_argument('--outfile', help='Output CSV filename. By default a filename based on the project and snapshot timestamp is used')
    return parser.parse_args()


def read_token(path):
    with open(path) as f:
        return f.read().strip()


def make_full_snapshot(ls, project, title=None):    
    """Create a full PROJECT snapshot."""
    print('Creating a project snapshot...')
    try:
        snap = ls.projects.exports.create(project_id=project.id, title=title)
    except httpx.ReadTimeout as e:
        #print(type(e), e)
        snaps = ls.projects.exports.list(project_id=project.id)
        #print(snaps)
        snap = [snap for snap in snaps if snap.title==title][0]
    return snap

def make_filtered_snapshot(ls, project, title, filter_obj):
    """Create a filtered project snapshot."""
    print('Creating a FILTERED snapshot....')
    filterview = ls.views.create(
        project=project.id,
        data= dict(
            filters = filter_obj,
            title = f'Snapshot FilterView for "{title}"')
    )
    
    try:
        snap = ls.projects.exports.create(project_id=project.id, title=title, 
                    task_filter_options = dict(view=filterview.id))
    except httpx.ReadTimeout as e:
        #print(type(e), e)
        snaps = ls.projects.exports.list(project_id=project.id)
        #print(snaps)
        snap = [snap for snap in snaps if snap.title==title][0]
    return snap

def make_snapshot(ls, project, title=None, filter_obj=None):
    """Create a snapshot with optional filtering."""
    if title is None:
        title = f'{project.title} at {dt.datetime.now().isoformat(timespec="seconds")}'
    
    if filter_obj:
        snap = make_filtered_snapshot(ls, project, title, filter_obj)
    else:
        snap = make_full_snapshot(ls, project, title)
    return snap

def wait_for_snapshot_completion(ls, project_id, snap):
    if isinstance(snap, int):
        snap = ls.projects.exports.get(project_id, snap)
    # Wait for snapshot to complete
    while snap.status in ["created", "in_progress"]:
        print(f'Snapshot status: {snap.id} "{snap.title}" - {snap.status}')
        snap = ls.projects.exports.get(project_id, snap.id)
        time.sleep(10)
    assert snap.status == 'completed', f'Snapshot status is "{snap.status}"'
    print(f'Snapshot status: {snap.id} "{snap.title}" - {snap.status}')
    return snap


def create_filter_obj_from_json(json_str):
    
    if os.path.isfile(json_str):
        with open(json_str) as f:
            filter_dict = json.load(f)
    else:
        filter_dict = json.loads(json_str)


    if filter_dict['conjunction'].lower() == 'or':
        operator = Filters.OR
    elif filter_dict['conjunction'].lower() == 'and':
        operator = Filters.AND
    else:
        raise ValueError("Invalid conjunction. Only 'and' and 'or' are allowed.")

    filter_items = []
    for item in filter_dict['items']:
        assert item['filter'].startswith('filter:tasks:data.')
        assert item['type'] in ['String']  # todo add more datatypes
        data_col = item['filter'].split(".",1)[1]
        operator = item['operator'].upper()
        dtype = item['type'].upper()
        value = item['value']
        filter_items.append(
            Filters.item(
                Column.data(data_col),
                getattr(Operator, operator),
                getattr(Type, item['type']),
                Filters.value(value)
            )
        )

    return Filters.create(operator, filter_items)



def download_snap_as_df(ls, project_id, snap_id):
    print('Downloading Snapshot as Dataframe...')
    chunks = ls.projects.exports.download(project_id, snap_id, export_type='CSV')

    csv_content = io.BytesIO()
    for chunk in chunks:
        csv_content.write(chunk)
    csv_content.seek(0)

    df = pd.read_csv(csv_content, low_memory=False)
    return df


def cleanup_df(df):
    print('Cleaning-up dataframe')
    # Define fields
    numerical_annot_fields = [
       'A1', 'A2', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 
       'E1', 'E2', 'E3', 'E4', 'F1', 'G1', 'G2', 'G3', 'H1',
       'I1', 'I2', 'I3', 'J1', 'J2', 'K1', 'K2', 'L1', 'L2', 'L3',
       'L4', 'M1', 'M2', 'M3', 'M4', 'N1', 'N2', 'N3',
       'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10']
    extra_annot_fields = ['Z1','Type']
    cache_fields = ['cache_'+col for col in extra_annot_fields+numerical_annot_fields]
    data_fields = ['pid', 'site', 'slide', 'sediment', 'iodp', 'fraction', 
                   'Slide Hole', 'Slide Hole Max', 'obj', 'magnification',
                   'Notes on Coding', 'Sample #', 'Object #', 
                   'Depth', 'core', 'section', 'interval', 'image']
    export_fields = ['id','annotation_id','annotator','created_at','updated_at']

    ordered_fields = export_fields + data_fields + extra_annot_fields + numerical_annot_fields

    # Note excluded fields
    excluded_fields = sorted(set(df.columns) - set(ordered_fields))
    if excluded_fields:
        #print(f"Excluded fields: {excluded_fields}")
        ...

    # Cleaning up taxonomies
    for col in numerical_annot_fields+extra_annot_fields:
        if col in df.columns:
            df[col] = df[col].str.split('"').str[3] if isinstance(df[col].iloc[0], str) else df[col]

    # Create a copy of the dataframe with ordered fields
    df = df[ordered_fields].copy()

    # Add numerical matrix
    for col in numerical_annot_fields:
        df[col+'.1'] = df[col].str.split('.').str[1]

    return df


def cleanup_snapshot(ls, project_id, snap, cleanup_filterview=True):

    try:
        ls.projects.exports.delete(project_id=project_id, export_pk=snap.id)
        print(f"DELETED: Snapshot {snap.id} deleted successfully")
    except Exception as e:
        print(type(e), f'Error deleting snapshot {snap.id} "{snap.title}"')
    
    if cleanup_filterview:
        view_title = f'Snapshot FilterView for "{snap.title}"'
        views = ls.views.list(project=project_id)
        try:
            view = [v for v in views if v.data['title']==view_title][0]
            ls.views.delete(view.id)
            print(f'DELETED: View {view.id} "{view_title}" deleted successfully')
        except Exception as e:
            print(type(e), f'View "{view_title}" failed to be removed')




def main():
    # Parse command-line arguments
    args = parse_arguments()
    token = read_token(args.token)

    # Connect to the Label Studio API
    ls = LabelStudio(base_url=args.host, api_key=token, timeout=10)

    # Get the project
    project = ls.projects.get(id=args.project)

    # Create filter if argument provided
    filter_obj = None
    if args.filter:
        filter_obj = create_filter_obj_from_json(args.filter)

    # Define a title for the snapshot
    title = args.outfile or f'{project.title} at {dt.datetime.now().isoformat(timespec="seconds")}'

    # Create a Label Studio snapshot
    if not args.snapshot:
        snap = make_snapshot(ls, project, title, filter_obj)
    snap = wait_for_snapshot_completion(ls, project.id, args.snapshot or snap)

    # Download snapshot into dataframe
    df = download_snap_as_df(ls, project.id, snap.id)

    df = cleanup_df(df)

    # Save CSV
    if args.outfile:
        outfile = args.outfile
    else:
        outfile = f'{title}.csv'
    output_path = os.path.join(args.outdir, outfile)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'DataFrame written to: "{output_path}"')

    # Cleanup snapshot object
    if not args.nocleanup:
        cleanup_snapshot(ls, project.id, snap, bool(filter_obj))



if __name__ == '__main__':
    main()
    

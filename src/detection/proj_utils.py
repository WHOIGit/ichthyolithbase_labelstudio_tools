import os
import shutil
from typing import Union

import numpy as np

from PIL import Image,ImageDraw
from label_studio_sdk.converter.brush import mask2rle


def coco_poly_to_ls_rle(poly_xy, image_width, image_height):
    mask_img = Image.new("L", (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask_img)

    xs = np.clip(np.array(poly_xy[0::2], dtype=np.float32), 0, image_width - 1)
    ys = np.clip(np.array(poly_xy[1::2], dtype=np.float32), 0, image_height - 1)
    pts = list(map(tuple, np.stack([xs, ys], axis=1)))
    draw.polygon(pts, outline=1, fill=1)
    mask = np.array(mask_img, dtype=np.uint8)

    # labelstudio rle
    rle = mask2rle(mask)
    return rle


from sahi.annotation import ObjectAnnotation, BoundingBox
from sahi.prediction import PredictionResult, ObjectPrediction

def sort_prediction_result_SCANLINE(opl:list[Union[ObjectAnnotation,ObjectPrediction]]):
    row_groups = []
    try:
        opl = sorted(opl, key=lambda obj:obj.bbox.miny) # order by top of bbox
        is_BoundingBox = True
    except AttributeError:
        opl = sorted(opl, key=lambda obj:obj.bbox[1])
        is_BoundingBox = False

    def objs_in_pixelrow(objs, y_px, assume_sorted=True):
        # todo assume objs is sorted, find efficiencies
        row_obj = []
        if is_BoundingBox:
            for obj in objs:
                if obj.bbox.miny < y_px <= obj.bbox.maxy:
                    row_obj.append(obj)
        else:
            for obj in objs:
                if obj.bbox[1] < y_px <= obj.bbox[1]+obj.bbox[3]:
                    row_obj.append(obj)
        return row_obj

    prev_row = []
    i=0
    while opl: # toto efficiency: advance by n pixels at a time, where n is ~height of smallest bbox
        i += 1
        candidate_row =  objs_in_pixelrow(opl, i)

        # if candidate row is missing an object that is in previous row
        if set(prev_row)-set(candidate_row):
            #    save_previous row to row_groups,
            if is_BoundingBox:
                row_groups.extend( sorted(prev_row, key=lambda obj:obj.bbox.minx) )
            else:
                row_groups.extend(sorted(prev_row, key=lambda obj: obj.bbox[0]))
            #    remove obj from opl (adds efficiency)
            opl = opl[len(prev_row):]
            #    remove captured objs from next prev_row
            prev_row = list(set(candidate_row)-set(prev_row))
        else:
            # candidate_row becomes next previous row
            prev_row = candidate_row
    return row_groups

def sort_prediction_result_CLUSTER(pr:PredictionResult, eps=0.5):
    from sklearn.cluster import DBSCAN

    def reading_order_rowwise(boxes):
        """
        boxes: iterable of (x_min, y_min, x_max, y_max)
        returns: list of indices in human reading order (0..N-1)
        """
        boxes = np.asarray(boxes)
        cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
        cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
        heights = (boxes[:, 3] - boxes[:, 1])
        median_h = np.median(heights)

        # Cluster by y using vertical tolerance ~ half the median height (tweak as needed)
        # eps is in *feature units*; we cluster only on y so we pass [[y], [y], ...]
        y_features = cy.reshape(-1, 1)
        row_labels = DBSCAN(eps=eps * median_h, min_samples=1, metric='euclidean').fit_predict(y_features)

        # Order rows by their vertical position (top to bottom)
        ordered_rows = sorted(
            np.unique(row_labels),
            key=lambda r: np.median(cy[row_labels == r])
        )

        # Within each row, sort by x (left to right)
        order = []
        for r in ordered_rows:
            idx = np.where(row_labels == r)[0]
            idx_sorted = idx[np.argsort(cx[idx])]
            order.extend(idx_sorted.tolist())
        return order

    boxes = [obj.bbox.box if isinstance(obj.bbox, BoundingBox) else obj.bbox  for obj in pr.object_prediction_list]
    boxes_order = reading_order_rowwise(boxes)
    pr.object_prediction_list = [pr.object_prediction_list[i] for i in boxes_order]
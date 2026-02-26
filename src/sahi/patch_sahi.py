import os
import time
import copy
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Union, Generator, Dict, Set

from sahi import BoundingBox
from sahi.utils.cv import cv2, np, apply_color_mask, Colors
import sahi.utils.file  # for modified Path
from sahi.prediction import ObjectPrediction

from proj_utils import sort_prediction_result_SCANLINE


class MyColors(Colors):
    def __init__(self, colors:list):
        hex = tuple(colors)
        self.palette = [self.hex_to_rgb("#" + c) for c in hex]
        self.n = len(self.palette)

def visualize_object_predictions(
    image: np.ndarray,
    object_prediction_list: list[ObjectPrediction],
    rect_th: Optional[int] = None,
    text_size: Optional[float] = None,
    text_th: Optional[int] = None,
    color: Optional[tuple] = None,
    colors: Optional[list[str]] = None,  # NEW!
    label_pattern: Optional[str] = None,  # NEW!
    hide_mask: Optional[bool] = False,  # NEW!
    output_dir: Optional[str] = None,
    file_name: Optional[str] = "prediction_visual",
    export_format: Optional[str] = "png",
):
    """
    Visualizes prediction category names, bounding boxes over the source image
    and exports it to output folder.

    Args:
        object_prediction_list: a list of prediction.ObjectPrediction
        rect_th: rectangle thickness
        text_size: size of the category name over box
        text_th: text thickness
        color: annotation color in the form: (0, 255, 0)
        colors: Override default label color list.
        label_pattern: format string pattern for label.
                       Recognizes "{LABEL}", "{CONF:.2f}", "{OBJID:03}".
                       E.g. "{LABEL:.3} {CONF:.2f}" will result in "cat 0.98" for LABEL="category", CONF=0.9765
        output_dir: directory for resulting visualization to be exported
        file_name: exported file will be saved as: output_dir+file_name+".png"
        export_format: can be specified as 'jpg' or 'png'
    """
    elapsed_time = time.time()
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)
    # select predefined classwise color palette if not specified
    if color is None and colors is None:
        colors = Colors()
    elif colors:
        colors = MyColors(colors)
    else:
        colors = None
    # set rect_th for boxes
    rect_th = rect_th or max(round(sum(image.shape) / 2 * 0.003), 2)
    # set text_th for category names
    text_th = text_th or max(rect_th - 1, 1)
    # set text_size for category names
    text_size = text_size or rect_th / 3

    # add masks or obb polygons to image if present
    for idx,object_prediction in enumerate(object_prediction_list):
        # deepcopy object_prediction_list so that original is not altered
        object_prediction = object_prediction.deepcopy()
        # arange label to be displayed
        label_kwargs = dict(LABEL=object_prediction.category.name, CONF=object_prediction.score.value, OBJID=idx+1)
        label = label_pattern.format(**label_kwargs)
        # set color
        if colors is not None:
            color = colors(object_prediction.category.id)
        # visualize masks or obb polygons if present
        has_mask = object_prediction.mask is not None
        is_obb_pred = False
        if has_mask and not hide_mask:
            segmentation = object_prediction.mask.segmentation
            if len(segmentation) == 1 and len(segmentation[0]) == 8:
                is_obb_pred = True

            if is_obb_pred:
                points = np.array(segmentation).reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(image, [points], isClosed=True, color=color or (0, 0, 0), thickness=rect_th)

                if label_pattern:
                    lowest_point = points[points[:, :, 1].argmax()][0]
                    box_width, box_height = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[0]
                    outside = lowest_point[1] - box_height - 3 >= 0
                    text_bg_point1 = (
                        lowest_point[0],
                        lowest_point[1] - box_height - 3 if outside else lowest_point[1] + 3,
                    )
                    text_bg_point2 = (lowest_point[0] + box_width, lowest_point[1])
                    cv2.rectangle(
                        image, text_bg_point1, text_bg_point2, color or (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA
                    )
                    cv2.putText(
                        image,
                        label,
                        (lowest_point[0], lowest_point[1] - 2 if outside else lowest_point[1] + box_height + 2),
                        0,
                        text_size,
                        (255, 255, 255),
                        thickness=text_th,
                    )
            else:
                # draw mask
                rgb_mask = apply_color_mask(object_prediction.mask.bool_mask, color or (0, 0, 0))
                image = cv2.addWeighted(image, 1, rgb_mask, 0.6, 0)

        # add bboxes to image if is_obb_pred=False
        if not is_obb_pred:
            bbox = object_prediction.bbox.to_xyxy()

            # set bbox points
            point1, point2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
            # visualize boxes
            cv2.rectangle(
                image,
                point1,
                point2,
                color=color or (0, 0, 0),
                thickness=rect_th,
            )

            if label_pattern:
                box_width, box_height = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[
                    0
                ]  # label width, height
                outside = point1[1] - box_height - 3 >= 0  # label fits outside box
                point2 = point1[0] + box_width, point1[1] - box_height - 3 if outside else point1[1] + box_height + 3
                # add bounding box text
                cv2.rectangle(image, point1, point2, color or (0, 0, 0), -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    image,
                    label,
                    (point1[0], point1[1] - 2 if outside else point1[1] + box_height + 2),
                    0,
                    text_size,
                    (255, 255, 255),
                    thickness=text_th,
                )

    # export if output_dir is present
    if output_dir is not None:
        # export image with predictions
        sahi.utils.file.Path(output_dir).mkdir(parents=True, exist_ok=True)
        # save inference result
        save_path = str(sahi.utils.file.Path(output_dir) / ((file_name or "") + "." + (export_format or "")))
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    elapsed_time = time.time() - elapsed_time
    return {"image": image, "elapsed_time": elapsed_time}




from sahi.utils.coco import load_json, get_imageid2annotationlist_mapping, CocoAnnotation, CocoPrediction, Thread, threading, Lock


class CocoPlus(sahi.utils.coco.Coco):
    @classmethod
    def from_coco_dict_or_path(
        cls,
        coco_dict_or_path: Union[Dict, str],
        image_dir: Optional[str] = None,
        remapping_dict: Optional[Dict] = None,
        ignore_negative_samples: bool = False,
        clip_bboxes_to_img_dims: bool = False,
        use_threads: bool = False,
        num_threads: int = 10,
    ):
        """
        Creates coco object from COCO formatted dict or COCO dataset file path.

        Args:
            coco_dict_or_path: dict/str or List[dict/str]
                COCO formatted dict or COCO dataset file path
                List of COCO formatted dict or COCO dataset file path
            image_dir: str
                Base file directory that contains dataset images. Required for merging and yolov5 conversion.
            remapping_dict: dict
                {1:0, 2:1} maps category id 1 to 0 and category id 2 to 1
            ignore_negative_samples: bool
                If True ignores images without annotations in all operations.
            clip_bboxes_to_img_dims: bool = False
                Limits bounding boxes to image dimensions.
            use_threads: bool = False
                Use threads when processing the json image list, defaults to False
            num_threads: int = 10
                Slice the image list to given number of chunks, defaults to 10

        Properties:
            images: list of CocoImage
            category_mapping: dict
        """
        # init coco object
        coco = cls(
            image_dir=image_dir,
            remapping_dict=remapping_dict,
            ignore_negative_samples=ignore_negative_samples,
            clip_bboxes_to_img_dims=clip_bboxes_to_img_dims,
        )

        if type(coco_dict_or_path) not in [str, dict]:
            raise TypeError("coco_dict_or_path should be a dict or str")

        # load coco dict if path is given
        if isinstance(coco_dict_or_path, str):
            coco_dict = load_json(coco_dict_or_path)
        else:
            coco_dict = coco_dict_or_path

        dict_size = len(coco_dict["images"])

        # arrange image id to annotation id mapping
        coco.add_categories_from_coco_category_list(coco_dict["categories"])
        image_id_to_annotation_list = get_imageid2annotationlist_mapping(coco_dict)

        # NEW!
        image_id_to_prediction_list: Dict = defaultdict(list)
        if 'predictions' in coco_dict:
            logger.debug("indexing coco dataset predictions...")
            for prediction in coco_dict["predictions"]:
                image_id = prediction["image_id"]
                image_id_to_prediction_list[image_id].append(prediction)

        category_mapping = coco.category_mapping

        # https://github.com/obss/sahi/issues/98
        image_id_set: Set = set()

        lock = Lock()

        def fill_image_id_set(start, finish, image_list, _image_id_set, _image_id_to_annotation_list, _coco, lock):
            for coco_image_dict in tqdm(
                image_list[start:finish], f"Loading coco annotations between {start} and {finish}"
            ):
                coco_image = CocoImage.from_coco_image_dict(coco_image_dict)
                image_id = coco_image_dict["id"]
                # https://github.com/obss/sahi/issues/98
                if image_id in _image_id_set:
                    print(f"duplicate image_id: {image_id}, will be ignored.")
                    continue
                else:
                    lock.acquire()
                    _image_id_set.add(image_id)
                    lock.release()

                # select annotations of the image
                annotation_list = _image_id_to_annotation_list[image_id]
                for coco_annotation_dict in annotation_list:
                    # apply category remapping if remapping_dict is provided
                    if _coco.remapping_dict is not None:
                        # apply category remapping (id:id)
                        category_id = _coco.remapping_dict[coco_annotation_dict["category_id"]]
                        # update category id
                        coco_annotation_dict["category_id"] = category_id
                    else:
                        category_id = coco_annotation_dict["category_id"]
                    # get category name (id:name)
                    category_name = category_mapping[category_id]
                    coco_annotation = CocoAnnotation.from_coco_annotation_dict(
                        category_name=category_name, annotation_dict=coco_annotation_dict
                    )
                    coco_image.add_annotation(coco_annotation)
                _coco.add_image(coco_image)

        chunk_size = dict_size / num_threads

        if use_threads is True:
            for i in range(num_threads):
                start = i * chunk_size
                finish = start + chunk_size
                if finish > dict_size:
                    finish = dict_size
                t = Thread(
                    target=fill_image_id_set,
                    args=(start, finish, coco_dict["images"], image_id_set, image_id_to_annotation_list, coco, lock),
                )
                t.start()

            main_thread = threading.currentThread()
            for t in threading.enumerate():
                if t is not main_thread:
                    t.join()

        else:
            for coco_image_dict in tqdm(coco_dict["images"], "Loading coco annotations"):
                coco_image = CocoImage.from_coco_image_dict(coco_image_dict)
                image_id = coco_image_dict["id"]
                # https://github.com/obss/sahi/issues/98
                if image_id in image_id_set:
                    print(f"duplicate image_id: {image_id}, will be ignored.")
                    continue
                else:
                    image_id_set.add(image_id)
                # select annotations of the image
                annotation_list = image_id_to_annotation_list[image_id]
                # TODO: coco_annotation_dict is of type CocoAnnotation according to how image_id_to_annotation_list
                # was created. Either image_id_to_annotation_list is not defined correctly or the following
                # loop is wrong as it expects a dict.
                for coco_annotation_dict in annotation_list:
                    # apply category remapping if remapping_dict is provided
                    if coco.remapping_dict is not None:
                        # apply category remapping (id:id)
                        category_id = coco.remapping_dict[coco_annotation_dict["category_id"]]
                        # update category id
                        coco_annotation_dict["category_id"] = category_id
                    else:
                        category_id = coco_annotation_dict["category_id"]
                    # get category name (id:name)
                    category_name = category_mapping[category_id]
                    coco_annotation = CocoAnnotation.from_coco_annotation_dict(
                        category_name=category_name, annotation_dict=coco_annotation_dict
                    )
                    if isinstance(coco_annotation.bbox, list) and not coco_annotation.segmentation:
                        coco_annotation.bbox = BoundingBox(coco_annotation.bbox)
                    coco_image.add_annotation(coco_annotation)

                # NEW!
                prediction_list = image_id_to_prediction_list[image_id]
                for coco_prediction_dict in prediction_list:
                    # apply category remapping if remapping_dict is provided
                    if coco.remapping_dict is not None:
                        # apply category remapping (id:id)
                        category_id = coco.remapping_dict[coco_prediction_dict["category_id"]]
                        # update category id
                        coco_prediction_dict["category_id"] = category_id
                    else:
                        category_id = coco_prediction_dict["category_id"]
                    # get category name (id:name)
                    category_name = category_mapping[category_id]
                    coco_prediction = CocoPrediction.from_coco_annotation_dict(
                        category_name=category_name, annotation_dict=coco_prediction_dict,
                        score=coco_prediction_dict['score'])
                    if coco_prediction.score is None: coco_prediction.score = coco_prediction_dict['score']
                    coco_image.add_prediction(coco_prediction)

                coco.add_image(coco_image)

        if clip_bboxes_to_img_dims:
            coco = coco.get_coco_with_clipped_bboxes()

        if 'name' in coco_dict:
            coco.name = coco_dict['name']
        return coco

    @property
    def json(self):
        return self.create_coco_dict()

    def create_coco_dict(self, ignore_negative_samples=False, image_id_setting="auto"):
        """
        Creates COCO dict with fields "images", "annotations", "categories", "predictions" (NEW!), "name" (NEW!)

        Arguments
        ---------
            images : List of CocoImage containing a list of CocoAnnotation
            categories : List of Dict
                COCO categories
            ignore_negative_samples : Bool
                If True, images without annotations are ignored
            image_id_setting: str
                how to assign image ids while exporting can be
                    auto --> will assign id from scratch (<CocoImage>.id will be ignored)
                    manual --> you will need to provide image ids in <CocoImage> instances (<CocoImage>.id can not be None)
        Returns
        -------
            coco_dict : Dict
                COCO dict with fields "images", "annotations", "categories"
        """
        # assertion of parameters
        if image_id_setting not in ["auto", "manual"]:
            raise ValueError("'image_id_setting' should be one of ['auto', 'manual']")

        # define accumulators
        image_index = 1
        annotation_id = 1
        prediction_id = 1
        coco_dict = dict(name=self.name, categories=self.json_categories, images=[], annotations=[], predictions=[])
        if not self.name: coco_dict.pop('name')

        for coco_image in self.images:
            # get coco annotations
            coco_annotations = coco_image.annotations
            coco_predictions = coco_image.predictions
            # get num annotations
            num_annotations = len(coco_annotations)
            num_predictions = len(coco_predictions)
            # if ignore_negative_samples is True and no annotations, skip image
            if ignore_negative_samples and num_annotations == 0 and num_predictions == 0:
                continue
            else:
                # get image_id
                if image_id_setting == "auto":
                    image_id = image_index
                    image_index += 1
                elif image_id_setting == "manual":
                    if coco_image.id is None:
                        raise ValueError("'coco_image.id' should be set manually when image_id_setting == 'manual'")
                    image_id = coco_image.id

                # create coco image object
                out_image = {
                    "height": coco_image.height,
                    "width": coco_image.width,
                    "id": image_id,
                    "file_name": coco_image.file_name,
                }
                coco_dict["images"].append(out_image)

                # do the same for image annotations
                for coco_annotation in coco_annotations:
                    # create coco annotation object
                    out_annotation = {
                        "iscrowd": 0,
                        "image_id": image_id,
                        "bbox": coco_annotation.bbox,
                        "segmentation": coco_annotation.segmentation,
                        "category_id": coco_annotation.category_id,
                        "id": annotation_id,
                        "area": coco_annotation.area,
                    }
                    coco_dict["annotations"].append(out_annotation)
                    # increment annotation id
                    annotation_id += 1

                for coco_prediction in coco_predictions:
                    # create coco annotation object
                    out_prediction = {
                        "iscrowd": 0,
                        "image_id": image_id,
                        "bbox": coco_prediction.bbox,
                        "segmentation": coco_prediction.segmentation,
                        "category_id": coco_prediction.category_id,
                        "id": prediction_id,
                        "area": coco_prediction.area,
                        "score": coco_prediction.score,
                    }
                    coco_dict["predictions"].append(out_prediction)
                    # increment prediction id
                    prediction_id += 1

        # return coco dict
        return coco_dict


from tqdm import tqdm
from PIL import Image
from sahi.utils.coco import Coco, CocoImage
from sahi.predict import logger, increment_path, list_files, get_video_reader, read_image_as_pil, save_json, \
    DetectionModel, AutoDetectionModel, get_sliced_prediction, get_prediction, crop_object_predictions, \
    LOW_MODEL_CONFIDENCE, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, save_pickle


def predict(
    detection_model: Optional[DetectionModel] = None,
    model_type: str = "ultralytics",
    model_path: Optional[str] = None,
    model_config_path: Optional[str] = None,
    model_confidence_threshold: float = 0.25,
    model_device: Optional[str] = None,
    model_category_mapping: Optional[dict] = None,
    model_category_remapping: Optional[dict] = None,
    source: Optional[str] = None,
    no_standard_prediction: bool = False,
    no_sliced_prediction: bool = False,
    image_size: Optional[int] = None,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
    novisual: bool = False,
    view_video: bool = False,
    frame_skip_interval: int = 0,
    export_pickle: bool = False,
    export_crop: bool = False,
    dataset_json_path: Optional[str] = None,
    project: str = "runs/predict",
    name: str = "exp",
    visual_bbox_thickness: Optional[int] = None,
    visual_text_size: Optional[float] = None,
    visual_text_thickness: Optional[int] = None,
    visual_colors: Optional[List[str]] = None,
    visual_coco_annotations_color = (0, 255, 0),  # blue
    visual_coco_predictions_color = (255, 0, 0),  # red
    visual_label_pattern: str = None,
    visual_export_format: str = "jpg",
    verbose: int = 1,
    return_dict: bool = False,
    force_postprocess_type: bool = False,
    exclude_classes_by_name: Optional[List[str]] = None,
    exclude_classes_by_id: Optional[List[int]] = None,
    **kwargs,
):
    """
    Performs prediction for all present images in given folder.

    Args:
        detection_model: sahi.model.DetectionModel
            Optionally provide custom DetectionModel to be used for inference. When provided,
            model_type, model_path, config_path, model_device, model_category_mapping, image_size
            params will be ignored
        model_type: str
            mmdet for 'MmdetDetectionModel', 'yolov5' for 'Yolov5DetectionModel'.
        model_path: str
            Path for the model weight
        model_config_path: str
            Path for the detection model config file
        model_confidence_threshold: float
            All predictions with score < model_confidence_threshold will be discarded.
        model_device: str
            Torch device, "cpu" or "cuda"
        model_category_mapping: dict
            Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
        model_category_remapping: dict: str to int
            Remap category ids after performing inference
        source: str
            Folder directory that contains images or path of the image to be predicted. Also video to be predicted.
        no_standard_prediction: bool
            Dont perform standard prediction. Default: False.
        no_sliced_prediction: bool
            Dont perform sliced prediction. Default: False.
        image_size: int
            Input image size for each inference (image is scaled by preserving asp. rat.).
        slice_height: int
            Height of each slice.  Defaults to ``512``.
        slice_width: int
            Width of each slice.  Defaults to ``512``.
        overlap_height_ratio: float
            Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        overlap_width_ratio: float
            Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        postprocess_type: str
            Type of the postprocess to be used after sliced inference while merging/eliminating predictions.
            Options are 'NMM', 'GREEDYNMM', 'LSNMS' or 'NMS'. Default is 'GREEDYNMM'.
        postprocess_match_metric: str
            Metric to be used during object prediction matching after sliced prediction.
            'IOU' for intersection over union, 'IOS' for intersection over smaller area.
        postprocess_match_threshold: float
            Sliced predictions having higher iou than postprocess_match_threshold will be
            postprocessed after sliced prediction.
        postprocess_class_agnostic: bool
            If True, postprocess will ignore category ids.
        novisual: bool
            Dont export predicted video/image visuals.
        view_video: bool
            View result of prediction during video inference.
        frame_skip_interval: int
            If view_video or export_visual is slow, you can process one frames of 3(for exp: --frame_skip_interval=3).
        export_pickle: bool
            Export predictions as .pickle
        export_crop: bool
            Export predictions as cropped images.
        dataset_json_path: str
            If coco file path is provided, detection results will be exported in coco json format.
        project: str
            Save results to project/name.
        name: str
            Save results to project/name.
        visual_bbox_thickness: int
        visual_text_size: float
        visual_text_thickness: int
        visual_colors: list[str]
        visual_label_pattern: str
            format string pattern for label.
            Recognizes "{LABEL}", "{CONF:.2f}", "{OBJID:03}".
            E.g. "{LABEL:.3} {CONF:.2f}" will result in "cat 0.98" for LABEL="category", CONF=0.9765
        visual_export_format: str
            Can be specified as 'jpg' or 'png'
        verbose: int
            0: no print
            1: print slice/prediction durations, number of slices
            2: print model loading/file exporting durations
        return_dict: bool
            If True, returns a dict with 'export_dir' field.
        force_postprocess_type: bool
            If True, auto postprocess check will e disabled
        exclude_classes_by_name: Optional[List[str]]
            None: if no classes are excluded
            List[str]: set of classes to exclude using its/their class label name/s
        exclude_classes_by_id: Optional[List[int]]
            None: if no classes are excluded
            List[int]: set of classes to exclude using one or more IDs
    """
    # assert prediction type
    if no_standard_prediction and no_sliced_prediction:
        raise ValueError("'no_standard_prediction' and 'no_sliced_prediction' cannot be True at the same time.")

    # auto postprocess type
    if not force_postprocess_type and model_confidence_threshold < LOW_MODEL_CONFIDENCE and postprocess_type != "NMS":
        logger.warning(
            f"Switching postprocess type/metric to NMS/IOU since confidence threshold is low ({model_confidence_threshold})."
        )
        postprocess_type = "NMS"
        postprocess_match_metric = "IOU"

    # for profiling
    durations_in_seconds = dict()

    # init export directories
    save_dir = Path(increment_path(Path(project) / name, exist_ok=False))  # increment run
    crop_dir = save_dir / "crops"
    visual_dir = save_dir / "visuals"
    visual_with_gt_dir = save_dir / "visuals_with_gt"
    pickle_dir = save_dir / "pickles"
    if not novisual or export_pickle or export_crop or dataset_json_path is not None:
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # init image iterator
    # TODO: rewrite this as iterator class as in https://github.com/ultralytics/yolov5/blob/d059d1da03aee9a3c0059895aa4c7c14b7f25a9e/utils/datasets.py#L178
    source_is_video = False
    num_frames = None
    image_iterator: Union[list[str], Generator[Image.Image, None, None]]
    if dataset_json_path and source:
        coco: Coco = Coco.from_coco_dict_or_path(dataset_json_path)
        image_iterator = [str(Path(source) / Path(coco_image.file_name)) for coco_image in coco.images]
        coco_json = []
    elif source and os.path.isdir(source):
        image_iterator = list_files(directory=source, contains=IMAGE_EXTENSIONS, verbose=verbose)
    elif source and Path(source).suffix in VIDEO_EXTENSIONS:
        source_is_video = True
        read_video_frame, output_video_writer, video_file_name, num_frames = get_video_reader(
            source, str(save_dir), frame_skip_interval, not novisual, view_video
        )
        image_iterator = read_video_frame
    elif source:
        image_iterator = [source]
    else:
        logger.error("No valid input given to predict function")
        return

    # init model instance
    time_start = time.time()
    if detection_model is None:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=model_path,
            config_path=model_config_path,
            confidence_threshold=model_confidence_threshold,
            device=model_device,
            category_mapping=model_category_mapping,
            category_remapping=model_category_remapping,
            load_at_init=False,
            image_size=image_size,
            **kwargs,
        )
        detection_model.load_model()
    time_end = time.time() - time_start
    durations_in_seconds["model_load"] = time_end

    # iterate over source images
    durations_in_seconds["prediction"] = 0
    durations_in_seconds["slice"] = 0

    input_type_str = "video frames" if source_is_video else "images"

    # NEW!
    coco_output = CocoPlus(name=str(save_dir))
    coco_output.add_categories_from_coco_category_list([dict(id=idx, name=label) for idx, label in detection_model.category_mapping.items()])
    return_object = dict(export_dir = save_dir, coco = coco_output)

    for ind, image_path in enumerate(
        tqdm(image_iterator, f"Performing inference on {input_type_str}", total=num_frames)
    ):
        # Source is an image: Iterating over Image objects
        if source and source_is_video:
            video_name = Path(source).stem
            relative_filepath = video_name + "_frame_" + str(ind)
        elif isinstance(image_path, Image.Image):
            raise RuntimeError("Source is not a video, but image is still an Image object ")
        # preserve source folder structure in export
        elif source and os.path.isdir(source):
            relative_filepath = str(Path(image_path)).split(str(Path(source)))[-1]
            relative_filepath = relative_filepath[1:] if relative_filepath[0] == os.sep else relative_filepath
        else:  # no process if source is single file
            relative_filepath = Path(image_path).name

        filename_without_extension = Path(relative_filepath).stem

        # load image
        image_as_pil = read_image_as_pil(image_path)

        # perform prediction
        if not no_sliced_prediction:
            # get sliced prediction
            prediction_result = get_sliced_prediction(
                image=image_as_pil,
                detection_model=detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                perform_standard_pred=not no_standard_prediction,
                postprocess_type=postprocess_type,
                postprocess_match_metric=postprocess_match_metric,
                postprocess_match_threshold=postprocess_match_threshold,
                postprocess_class_agnostic=postprocess_class_agnostic,
                verbose=1 if verbose else 0,
                exclude_classes_by_name=exclude_classes_by_name,
                exclude_classes_by_id=exclude_classes_by_id,
            )
            object_prediction_list = prediction_result.object_prediction_list
            if prediction_result.durations_in_seconds:
                durations_in_seconds["slice"] += prediction_result.durations_in_seconds["slice"]
        else:
            # get standard prediction
            prediction_result = get_prediction(
                image=image_as_pil,
                detection_model=detection_model,
                shift_amount=[0, 0],
                full_shape=None,
                postprocess=None,
                verbose=0,
                exclude_classes_by_name=exclude_classes_by_name,
                exclude_classes_by_id=exclude_classes_by_id,
            )
            object_prediction_list = prediction_result.object_prediction_list

        # NEW! Prediction list gets sorted top-to-bottom, left-to-right
        #      And coco image gets constructed and added to coco_predictons for output
        object_prediction_list = sort_prediction_result_SCANLINE(object_prediction_list)
        coco_image = CocoImage(file_name=relative_filepath, id=ind,
            width=image_as_pil.size[0], height=image_as_pil.size[1])
        for object_prediction in object_prediction_list:
            coco_image.add_prediction(object_prediction.to_coco_prediction(image_id=coco_image.id))
        coco_output.add_image(coco_image)

        durations_in_seconds["prediction"] += prediction_result.durations_in_seconds["prediction"]
        # Show prediction time
        if verbose:
            tqdm.write(
                "Prediction time is: {:.2f} ms".format(prediction_result.durations_in_seconds["prediction"] * 1000)
            )

        if dataset_json_path:
            if source_is_video is True:
                raise NotImplementedError("Video input type not supported with coco formatted dataset json")

            # append predictions in coco format
            for object_prediction in object_prediction_list:
                coco_prediction = object_prediction.to_coco_prediction()
                coco_prediction.image_id = coco.images[ind].id
                coco_prediction_json = coco_prediction.json
                if coco_prediction_json["bbox"]:
                    coco_json.append(coco_prediction_json)
            if not novisual:
                # convert ground truth annotations to object_prediction_list
                coco_image: CocoImage = coco.images[ind]
                object_prediction_gt_list: List[ObjectPrediction] = []
                for coco_annotation in coco_image.annotations:
                    coco_annotation_dict = coco_annotation.json
                    category_name = coco_annotation.category_name
                    full_shape = [coco_image.height, coco_image.width]
                    object_prediction_gt = ObjectPrediction.from_coco_annotation_dict(
                        annotation_dict=coco_annotation_dict, category_name=category_name, full_shape=full_shape
                    )
                    object_prediction_gt_list.append(object_prediction_gt)
                # export visualizations with ground truths
                output_dir = str(visual_with_gt_dir / Path(relative_filepath).parent)
                result = visualize_object_predictions(
                    np.ascontiguousarray(image_as_pil),
                    object_prediction_list=object_prediction_gt_list,
                    rect_th=visual_bbox_thickness,
                    text_size=visual_text_size,
                    text_th=visual_text_thickness,
                    color=visual_coco_annotations_color,
                    label_pattern=visual_label_pattern,
                    output_dir=None,
                    file_name=None,
                    export_format=None,
                )
                _ = visualize_object_predictions(
                    result["image"],
                    object_prediction_list=object_prediction_list,
                    rect_th=visual_bbox_thickness,
                    text_size=visual_text_size,
                    text_th=visual_text_thickness,
                    color=visual_coco_predictions_color,
                    label_pattern=visual_label_pattern,
                    output_dir=output_dir,
                    file_name=filename_without_extension,
                    export_format=visual_export_format,
                )

        time_start = time.time()
        # export prediction boxes
        if export_crop:
            output_dir = str(crop_dir / Path(relative_filepath).parent)
            crop_object_predictions(
                image=np.ascontiguousarray(image_as_pil),
                object_prediction_list=object_prediction_list,
                output_dir=output_dir,
                file_name=filename_without_extension,
                export_format=visual_export_format,
            )
        # export prediction list as pickle
        if export_pickle:
            save_path = str(pickle_dir / Path(relative_filepath).parent / (filename_without_extension + ".pickle"))
            save_pickle(data=object_prediction_list, save_path=save_path)

        # export visualization
        if not novisual or view_video:
            output_dir = str(visual_dir / Path(relative_filepath).parent)
            result = visualize_object_predictions(
                np.ascontiguousarray(image_as_pil),
                object_prediction_list=object_prediction_list,
                rect_th=visual_bbox_thickness,
                text_size=visual_text_size,
                text_th=visual_text_thickness,
                label_pattern=visual_label_pattern,
                colors=visual_colors,
                output_dir=output_dir if not source_is_video else None,
                file_name=filename_without_extension,
                export_format=visual_export_format,
            )
            if not novisual and source_is_video:  # export video
                if output_video_writer is None:
                    raise RuntimeError("Output video writer could not be created")
                output_video_writer.write(cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR))

        # render video inference
        if view_video:
            cv2.imshow("Prediction of {}".format(str(video_file_name)), result["image"])
            cv2.waitKey(1)

        time_end = time.time() - time_start
        durations_in_seconds["export_files"] = time_end

    # export coco results
    if dataset_json_path:
        save_path = str(save_dir / "result.json")
        save_json(coco_json, save_path)

    if not novisual or export_pickle or export_crop or dataset_json_path is not None:
        print(f"Prediction results are successfully exported to {save_dir}")

    # print prediction duration
    if verbose == 2:
        print(
            "Model loaded in",
            durations_in_seconds["model_load"],
            "seconds.",
        )
        print(
            "Slicing performed in",
            durations_in_seconds["slice"],
            "seconds.",
        )
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )
        if not novisual:
            print(
                "Exporting performed in",
                durations_in_seconds["export_files"],
                "seconds.",
            )

    if return_dict:
        return return_object






import sahi.utils.coco

export_single_yolo_image_and_corresponding_txt_box = sahi.utils.coco.export_single_yolo_image_and_corresponding_txt

def export_single_yolo_image_and_corresponding_txt_seg(
    coco_image, coco_image_dir, output_dir, ignore_negative_samples=False, disable_symlink=False
):
    """
    Generates YOLO formatted image symlink and polygon segmentation annotation txt file.

    Args:
        coco_image: sahi.utils.coco.CocoImage
        coco_image_dir: str
        output_dir: str
            Export directory.
        ignore_negative_samples: bool
            If True ignores images without annotations in all operations.
    """
    # if coco_image contains any invalid annotations, skip it
    contains_invalid_annotations = False
    for coco_annotation in coco_image.annotations:
        if len(coco_annotation.bbox) != 4:
            contains_invalid_annotations = True
            break
    if contains_invalid_annotations:
        return
    # skip images without annotations
    if len(coco_image.annotations) == 0 and ignore_negative_samples:
        return
    # skip images without suffix
    # https://github.com/obss/sahi/issues/114
    if Path(coco_image.file_name).suffix == "":
        print(f"image file has no suffix, skipping it: '{coco_image.file_name}'")
        return
    elif Path(coco_image.file_name).suffix in [".txt"]:  # TODO: extend this list
        print(f"image file has incorrect suffix, skipping it: '{coco_image.file_name}'")
        return
    # set coco and yolo image paths
    if Path(coco_image.file_name).is_file():
        coco_image_path = os.path.abspath(coco_image.file_name)
    else:
        if coco_image_dir is None:
            raise ValueError("You have to specify image_dir of Coco object for yolo conversion.")

        coco_image_path = os.path.abspath(str(Path(coco_image_dir) / coco_image.file_name))

    yolo_image_path_temp = str(Path(output_dir) / Path(coco_image.file_name).name)
    # increment target file name if already present
    yolo_image_path = copy.deepcopy(yolo_image_path_temp)
    name_increment = 2
    while Path(yolo_image_path).is_file():
        parent_dir = Path(yolo_image_path_temp).parent
        filename = Path(yolo_image_path_temp).stem
        filesuffix = Path(yolo_image_path_temp).suffix
        filename = filename + "_" + str(name_increment)
        yolo_image_path = str(parent_dir / (filename + filesuffix))
        name_increment += 1
    # create a symbolic link pointing to coco_image_path named yolo_image_path
    if disable_symlink:
        import shutil

        shutil.copy(coco_image_path, yolo_image_path)
    else:
        os.symlink(coco_image_path, yolo_image_path)
    # calculate annotation normalization ratios
    width = coco_image.width
    height = coco_image.height
    dw = 1.0 / (width)
    dh = 1.0 / (height)
    # set annotation filepath
    image_file_suffix = Path(yolo_image_path).suffix
    yolo_annotation_path = yolo_image_path.replace(image_file_suffix, ".txt")
    # create annotation file
    annotations = coco_image.annotations
    with open(yolo_annotation_path, "w") as outfile:
        for annotation in annotations:
            # convert coco bbox to yolo bbox
            yolo_poly = []
            coco_poly = annotation.segmentation[0]  # doesn't handle multi-area segment polygons
            for i in range(0, len(coco_poly), 2):
                 x = coco_poly[i] * dw
                 y = coco_poly[i + 1] * dh
                 yolo_poly.extend([x, y])

            category_id = annotation.category_id
            # save yolo annotation
            outfile.write(str(category_id) + " " + " ".join([str(value) for value in yolo_poly]) + "\n")




from sahi.slicing import SliceImageResult, get_slice_bboxes, IMAGE_EXTENSIONS_LOSSY, IMAGE_EXTENSIONS_LOSSLESS, \
    process_coco_annotations, SlicedImage, concurrent, MAX_WORKERS, TopologicalError

def slice_image(
    image: Union[str, Image.Image, CocoImage],
    image_dir: Optional[str] = '.',
    output_file_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    slice_height: Optional[int] = None,
    slice_width: Optional[int] = None,
    overlap_height_ratio: Optional[float] = 0.2,
    overlap_width_ratio: Optional[float] = 0.2,
    auto_slice_resolution: Optional[bool] = True,
    min_area_ratio: Optional[float] = 0.1,
    out_ext: Optional[str] = None,
    verbose: Optional[bool] = False,
    exif_fix: bool = True,
    clobber: bool = False,
) -> SliceImageResult:
    """Slice a large image into smaller windows. If output_file_name and output_dir is given, export
    sliced images.

    Args:
        image (str or PIL.Image or CocoImage): File path of image or Pillow Image to be sliced.
        image_dir (str): ...
        output_file_name (str, optional): Root name of output files (coordinates will
            be appended to this)
        output_dir (str, optional): Output directory
        slice_height (int, optional): Height of each slice. Default None.
        slice_width (int, optional): Width of each slice. Default None.
        overlap_height_ratio (float, optional): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio (float, optional): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        auto_slice_resolution (bool, optional): if not set slice parameters such as slice_height and slice_width,
            it enables automatically calculate these params from image resolution and orientation.
        min_area_ratio (float, optional): If the cropped annotation area to original annotation
            ratio is smaller than this value, the annotation is filtered out. Default 0.1.
        out_ext (str, optional): Extension of saved images. Default is the
            original suffix for lossless image formats and png for lossy formats ('.jpg','.jpeg').
        verbose (bool, optional): Switch to print relevant values to screen.
            Default 'False'.
        exif_fix (bool): Whether to apply an EXIF fix to the image.

    Returns:
        sliced_image_result: SliceImageResult:
                                sliced_image_list: list of SlicedImage
                                image_dir: str
                                    Directory of the sliced image exports.
                                original_image_size: list of int
                                    Size of the unsliced original image in [height, width]
    """

    # define verboseprint
    verboselog = logger.info if verbose else lambda *a, **k: None

    def _export_single_slice(image: np.ndarray, output_dir: str, slice_file_name: str, clobber:bool=True):
        slice_file_path = str(Path(output_dir) / slice_file_name)
        if os.path.isfile(slice_file_path) and clobber is False:
            verboselog("sliced image path already exists: " + slice_file_path)
            return
        image_pil = read_image_as_pil(image, exif_fix=exif_fix)
        # export sliced image
        image_pil.save(slice_file_path)
        image_pil.close()  # to fix https://github.com/obss/sahi/issues/565
        verboselog("sliced image path: " + slice_file_path)

    # create outdir if not present
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # read image
    if isinstance(image, CocoImage):
        image_path: str = os.path.join(image_dir, image.file_name)
        image_pil = read_image_as_pil(image_path, exif_fix=exif_fix)
    else:
        image_pil = read_image_as_pil(image, exif_fix=exif_fix)
    verboselog("image.shape: " + str(image_pil.size))

    image_width, image_height = image_pil.size
    if not (image_width != 0 and image_height != 0):
        raise RuntimeError(f"invalid image size: {image_pil.size} for 'slice_image'.")
    slice_bboxes = get_slice_bboxes(
        image_height=image_height,
        image_width=image_width,
        auto_slice_resolution=auto_slice_resolution,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    n_ims = 0

    # init images and annotations lists
    sliced_image_result = SliceImageResult(original_image_size=[image_height, image_width], image_dir=output_dir)

    image_pil_arr = np.asarray(image_pil)
    # iterate over slices
    for slice_bbox in slice_bboxes:
        n_ims += 1

        # extract image
        tlx = slice_bbox[0]
        tly = slice_bbox[1]
        brx = slice_bbox[2]
        bry = slice_bbox[3]
        image_pil_slice = image_pil_arr[tly:bry, tlx:brx]

        # set image file suffixes
        slice_suffixes = "_".join(map(str, slice_bbox))
        if out_ext:
            suffix = out_ext
        elif hasattr(image_pil, "filename"):
            suffix = Path(getattr(image_pil, "filename")).suffix
            if suffix in IMAGE_EXTENSIONS_LOSSY:
                suffix = ".png"
            elif suffix in IMAGE_EXTENSIONS_LOSSLESS:
                suffix = Path(image_pil.filename).suffix
        else:
            suffix = ".png"

        # set image file name and path
        slice_file_name = f"{output_file_name}_{slice_suffixes}{suffix}"

        # create coco image
        slice_width = slice_bbox[2] - slice_bbox[0]
        slice_height = slice_bbox[3] - slice_bbox[1]
        coco_image_slice = CocoImage(file_name=slice_file_name, height=slice_height, width=slice_width)

        # append coco annotations (if present) to coco image
        if isinstance(image, CocoImage):
            if image.annotations is not None:
                for sliced_coco_annotation in process_coco_annotations(image.annotations, slice_bbox, min_area_ratio):
                    coco_image_slice.add_annotation(sliced_coco_annotation)

            # append coco predictions (if present) to coco image
            if image.predictions is not None:
                for sliced_coco_prediction in process_coco_annotations(image.predictions, slice_bbox, min_area_ratio):
                    sliced_coco_prediction = CocoPrediction.from_coco_annotation_dict(
                        category_name = sliced_coco_prediction.category_name,
                        annotation_dict = sliced_coco_prediction.json,
                        score = None,
                        image_id= sliced_coco_prediction.image_id)
                    coco_image_slice.add_prediction(sliced_coco_prediction)

        # create sliced image and append to sliced_image_result
        sliced_image = SlicedImage(
            image=image_pil_slice, coco_image=coco_image_slice, starting_pixel=[slice_bbox[0], slice_bbox[1]]
        )
        sliced_image_result.add_sliced_image(sliced_image)

    # export slices if output directory is provided
    if output_file_name and output_dir:
        conc_exec = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
        conc_exec.map(
            _export_single_slice,
            sliced_image_result.images,
            [output_dir] * len(sliced_image_result),
            sliced_image_result.filenames,
            [clobber] * len(sliced_image_result),
        )

    verboselog(
        "Num slices: " + str(n_ims) + " slice_height: " + str(slice_height) + " slice_width: " + str(slice_width)
    )

    return sliced_image_result




def slice_coco(
    coco_dict_or_path: Union[dict,str,CocoPlus],
    image_dir: str,
    output_dir: Optional[str] = None,
    slice_height: Optional[int] = 512,
    slice_width: Optional[int] = 512,
    overlap_height_ratio: Optional[float] = 0.2,
    overlap_width_ratio: Optional[float] = 0.2,
    min_area_ratio: Optional[float] = 0.1,
    out_ext: Optional[str] = None,
    verbose: Optional[bool] = False,
    exif_fix: bool = True,
    clobber: bool = False,
) -> (CocoPlus,dict):
    """
    Slice large images given in a directory, into smaller windows. If output_dir is given, export sliced images and coco file.

    Args:
        coco_dict_or_path (str): Location of the coco annotation file
        image_dir (str): Base directory for the images
        output_coco_annotation_file_name (str): File name of the exported coco
            dataset json.
        output_dir (str, optional): Output directory
        ignore_negative_samples (bool, optional): If True, images without annotations
            are ignored. Defaults to False.
        slice_height (int, optional): Height of each slice. Default 512.
        slice_width (int, optional): Width of each slice. Default 512.
        overlap_height_ratio (float, optional): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio (float, optional): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        min_area_ratio (float): If the cropped annotation area to original annotation
            ratio is smaller than this value, the annotation is filtered out. Default 0.1.
        out_ext (str, optional): Extension of saved images. Default is the
            original suffix.
        verbose (bool, optional): Switch to print relevant values to screen.
        exif_fix (bool, optional): Whether to apply an EXIF fix to the image.

    Returns:
        coco_obj, original_size_mapper: CocoPlus, dict

    """
    if isinstance(coco_dict_or_path, Coco):
        coco = coco_dict_or_path
    else:
        coco = CocoPlus.from_coco_dict_or_path(coco_dict_or_path)
    # create coco obj
    sliced_coco = CocoPlus(image_dir=output_dir)
    sliced_coco.add_categories_from_coco_category_list(coco.json_categories)

    # iterate over images and slice
    original_size_mapper = {}
    for idx, coco_image in enumerate(tqdm(coco.images)):
        # slice image
        try:
            slice_image_result = slice_image(
                image=coco_image,
                image_dir=image_dir,
                output_file_name=f"{Path(coco_image.file_name).stem}_",
                output_dir=output_dir,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                min_area_ratio=min_area_ratio,
                out_ext=out_ext,
                verbose=verbose,
                exif_fix=exif_fix,
                clobber=clobber,
            )
            # append slice outputs
            for image in slice_image_result.sliced_image_list:
                sliced_coco.add_image(image.coco_image)
                original_size_mapper[image.coco_image.file_name] = dict(
                    w=slice_image_result.original_image_width, h=slice_image_result.original_image_height)
        except TopologicalError:
            logger.warning(f"Invalid annotation found, skipping this image: {coco_image}")

    return sliced_coco, original_size_mapper







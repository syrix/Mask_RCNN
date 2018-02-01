from operator import attrgetter
import time
import math

import keras
import numpy as np
from recordclass import recordclass


class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, config, generate_validation_generator, num_images):
        super(keras.callbacks.Callback, self).__init__()

        self.num_classes = config.NUM_CLASSES
        self.generate_validation_generator = generate_validation_generator

        batch_size = config.BATCH_SIZE
        num_images = min(num_images, config.MAX_METRICS_IMAGES)
        self.num_steps = math.ceil(num_images / batch_size)
        print(f'Creating metrics callback for {self.num_steps * batch_size} validation images.')

    def on_epoch_end(self, epoch, logs={}):
        metrics = Metrics(self.num_classes)
        generator = self.generate_validation_generator()

        for _ in range(self.num_steps):
            inputs, outputs = generator.__next__()

            images = inputs[0]
            image_meta = inputs[1]
            rpn_match = inputs[2]
            rpn_bbox = inputs[3]
            gt_class_ids = inputs[4]
            gt_boxes = inputs[5]
            gt_masks = inputs[6]

            y_pred = self.model.predict(inputs, verbose=0)
            detections = y_pred[14]

            for i in range(len(detections)):
                new_metrics = self._calculate_metrics(
                    pred_boxes=detections[i], gt_boxes=gt_boxes[i], gt_class_ids=gt_class_ids[i])
                metrics.merge(new_metrics)
        metrics.add_to_log(logs)

    def _calculate_metrics(self, pred_boxes, gt_boxes, gt_class_ids):
        """
        Calculate box-wise IoU for all classes in a single image
        :param pred_boxes: list of predicted boxes.
        Each box is a list of [min_y, min_x, max_y, max_x, class_id, score] as floats
        :param gt_boxes: list of grount truth boxes.
        Each box is a list of [min_y, min_x, max_y, max_x] as integers
        :param gt_class_ids: a list of ground truth class ids.
        :return: a dict with metrics per class
        """
        pred_box_class_id_index = 4

        # trim empty boxes
        pred_boxes = _trim_empty(pred_boxes)
        gt_boxes = _trim_empty(gt_boxes)
        gt_class_ids = gt_class_ids[0:len(gt_boxes)]

        # group ground truth boxes by class id
        gt_boxes_by_class = []
        for class_id in range(self.num_classes):
            current_boxes = []
            for i, box in enumerate(gt_boxes):
                if gt_class_ids[i] == class_id:
                    box = _convert_box(box)
                    current_boxes.append(box)
            gt_boxes_by_class.append(current_boxes)

        # group predicted boxes by class id
        pred_boxes_by_class = []
        for class_id in range(self.num_classes):
            current_boxes = []
            for box in pred_boxes:
                if box[pred_box_class_id_index] == class_id:
                    box = _convert_box(box)
                    current_boxes.append(box)
            pred_boxes_by_class.append(current_boxes)

        # match predicted to ground truth boxes
        metrics = Metrics(self.num_classes)
        for class_id in range(self.num_classes):
            overlaps = _find_overlaps(gt_boxes_by_class[class_id], pred_boxes_by_class[class_id])
            ious, false_positive_pixels, false_negative_pixels = \
                _match_boxes(gt_boxes_by_class[class_id], pred_boxes_by_class[class_id], overlaps)

            metrics.add(class_id, ious, false_positive_pixels, false_negative_pixels)
        return metrics


# TODO num unmatched pred_boxes, num matched boxes, num unmatched gt_boxes?
def _trim_empty(boxes):
    num_values_per_box = len(boxes[0])
    return np.array([box for box in boxes if np.all(box != ([0] * num_values_per_box))])


def _convert_box(box):
    Box = recordclass('Box', ['min_y', 'min_x', 'max_y', 'max_x'])
    return Box(min_y=box[0], min_x=box[1], max_y=box[2], max_x=box[3])


def _find_overlaps(gt_boxes, pred_boxes):
    overlaps = []
    for gt_index, gt_box in enumerate(gt_boxes):
        for pred_index, pred_box in enumerate(pred_boxes):
            Overlap = recordclass('Overlap',
                                  ['min_y', 'min_x', 'max_y', 'max_x', 'gt_index', 'pred_index',
                                   'iou', 'false_positive_pixels', 'false_negative_pixels'])
            overlap = Overlap(min_y=max(gt_box.min_y, pred_box.min_y),
                              min_x=max(gt_box.min_x, pred_box.min_x),
                              max_y=min(gt_box.max_y, pred_box.max_y),
                              max_x=min(gt_box.max_x, pred_box.max_x),
                              gt_index=gt_index, pred_index=pred_index,
                              iou=0, false_positive_pixels=0, false_negative_pixels=0)
            # Note that max_x and max_y are not part of the box => we need '>' not '>='
            if overlap.max_y > overlap.min_y and overlap.max_x > overlap.min_x:
                shared_area = (overlap.max_y - overlap.min_y) * (overlap.max_x - overlap.min_x)

                gt_area = (gt_box.max_y - gt_box.min_y) * (gt_box.max_x - gt_box.min_x)
                pred_area = (pred_box.max_y - pred_box.min_y) * (pred_box.max_x - pred_box.min_x)
                total_area = gt_area + pred_area - shared_area

                overlap.iou = shared_area / total_area
                # false positive pixels are pixels that were predicted, but are not in the ground truth
                overlap.false_positive_pixels = pred_area - shared_area
                # false negative pixels are pixels that are in the ground truth, but were not predicted
                overlap.false_negative_pixels = gt_area - shared_area
                overlaps.append(overlap)
    overlaps = sorted(overlaps, key=attrgetter('iou'), reverse=True)
    return overlaps


def _match_boxes(gt_boxes, pred_boxes, overlaps):
    """
    Greedily (highest iou first) match gt_boxes with pred_boxes and output one iou value per pred_box and
    additional zeroes for every non-matched gt_box
    """
    gt_matched = [False] * len(gt_boxes)
    pred_matched = [False] * len(pred_boxes)

    # Greedily assign the best matching box to each predicted box. Non-matched boxes get iou of 0
    ious = [0] * len(pred_boxes)
    false_positive_pixels = [-1] * len(pred_boxes)
    false_negative_pixels = [0] * len(pred_boxes)
    for overlap in overlaps:
        if not gt_matched[overlap.gt_index] and not pred_matched[overlap.pred_index]:
            gt_matched[overlap.gt_index] = True
            pred_matched[overlap.pred_index] = True
            ious[overlap.pred_index] = overlap.iou
            false_positive_pixels[overlap.pred_index] = overlap.false_positive_pixels
            false_negative_pixels[overlap.pred_index] = overlap.false_negative_pixels

    # Append an iou of 0 and the full area as FN-pixels for each non-matched ground truth box
    for gt_index, gt_box in enumerate(gt_boxes):
        if not gt_matched[gt_index]:
            ious.append(0)
            area = (gt_box.max_y - gt_box.min_y) * (gt_box.max_x - gt_box.min_x)
            false_negative_pixels.append(area)

    # Set the FP-pixels to the whole are of each non-matched predicted box
    for pred_index, pred_box in enumerate(pred_boxes):
        if not pred_matched[pred_index]:
            area = (pred_box.max_y - pred_box.min_y) * (pred_box.max_x - pred_box.min_x)
            false_positive_pixels[pred_index] = area

    return ious, false_positive_pixels, false_negative_pixels


class Metrics:
    def __init__(self, num_classes):
        self.start_time = time.time()
        self.ious = [[] for i in range(num_classes)]
        self.false_positive_pixels = [[] for i in range(num_classes)]
        self.false_negative_pixels = [[] for i in range(num_classes)]
        self.num_classes = num_classes

    def add(self, class_id, ious, false_positive_pixels, false_negative_pixels):
        self.ious[class_id] += ious
        self.false_positive_pixels[class_id] += false_positive_pixels
        self.false_negative_pixels[class_id] += false_negative_pixels

    def merge(self, metrics_object):
        assert(self.num_classes == metrics_object.num_classes)
        for class_id in range(self.num_classes):
            self.ious[class_id] += metrics_object.ious[class_id]
            self.false_positive_pixels[class_id] += metrics_object.false_positive_pixels[class_id]
            self.false_negative_pixels[class_id] += metrics_object.false_negative_pixels[class_id]

    def summarize(self):
        time_passed = time.time() - self.start_time
        print(f'done calculating metrics [time: {time_passed}]')
        for class_id in range(self.num_classes):
            print(f'mean iou for class {class_id}: '
                  f'{mean_or_zero(self.ious[class_id])}')
            print(f'mean false positive pixels for class {class_id}: '
                  f'{mean_or_zero(self.false_positive_pixels[class_id])}')
            print(f'mean false negative pixels for class {class_id}: '
                  f'{mean_or_zero(self.false_negative_pixels[class_id])}')

    def add_to_log(self, logs):
        for class_id in range(self.num_classes):
            logs[f'mean_iou_{class_id}'] = np.array(mean_or_zero(self.ious[class_id]))
            logs[f'mean_fpp_{class_id}'] = np.array(mean_or_zero(self.false_positive_pixels[class_id]))
            logs[f'mean_fnp_{class_id}'] = np.array(mean_or_zero(self.false_negative_pixels[class_id]))


def mean_or_zero(values):
    if values:
        return np.mean(values)
    return 0

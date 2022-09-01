# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
import numpy as np
from typing import List, Tuple

from PIL import Image, ImageDraw

from src.utils import common

class Object(collections.namedtuple('Object', ['id', 'score', 'bbox'])):
    """Represents a detected object.
    .. py:attribute:: id
        The object's class id.
    .. py:attribute:: score
        The object's prediction score.
    .. py:attribute:: bbox
        A :obj:`BBox` object defining the object's location.
    """

    @staticmethod
    def from_nparray(nparray, input_size=None, img_scale=(1.,1.)):
        return Object(
            id=int(nparray[0]),
            score=nparray[1],
            bbox=BBox.from_nparray(nparray[2:6], input_size, img_scale))

    @property
    def nparray(self):
        return np.concatenate(([self.id, self.score], self.bbox))

    def print(self, labels={}):
        print(labels.get(self.id, self.id))
        print('  id:    ', self.id)
        print('  score: ', self.score)
        print('  bbox:  ', self.bbox)


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """The bounding box for a detected object.
    .. py:attribute:: xmin
        X-axis start point
    .. py:attribute:: ymin
        Y-axis start point
    .. py:attribute:: xmax
        X-axis end point
    .. py:attribute:: ymax
        Y-axis end point
    """
    __slots__ = ()

    @staticmethod
    def from_nparray(nparray, input_size=None, img_scale=(1.,1.)):
        ymin, xmin, ymax, xmax = nparray
        bbox = BBox(xmin, ymin, xmax, ymax)
        if input_size is not None:
            width, height = input_size
            img_scale_x, img_scale_y = img_scale
            sx, sy = width / img_scale_x, height / img_scale_y
            bbox = bbox.scale(sx, sy).map(int)
        return bbox

    @property
    def nparray(self):
        return np.array(self)

    @property
    def width(self):
        """The bounding box width."""
        return self.xmax - self.xmin

    @property
    def height(self):
        """The bounding box height."""
        return self.ymax - self.ymin

    @property
    def area(self):
        """The bound box area."""
        return self.width * self.height

    @property
    def valid(self):
        """Indicates whether bounding box is valid or not (boolean).
        A valid bounding box has xmin <= xmax and ymin <= ymax (equivalent
        to width >= 0 and height >= 0).
        """
        return self.width >= 0 and self.height >= 0

    def scale(self, sx, sy):
        """Scales the bounding box.
        Args:
          sx (float): Scale factor for the x-axis.
          sy (float): Scale factor for the y-axis.
        Returns:
          A :obj:`BBox` object with the rescaled dimensions.
        """
        return BBox(xmin=sx * self.xmin,
                    ymin=sy * self.ymin,
                    xmax=sx * self.xmax,
                    ymax=sy * self.ymax)

    def translate(self, dx, dy):
        """Translates the bounding box position.
        Args:
          dx (int): Number of pixels to move the box on the x-axis.
          dy (int): Number of pixels to move the box on the y-axis.
        Returns:
          A :obj:`BBox` object at the new position.
        """
        return BBox(xmin=dx + self.xmin,
                    ymin=dy + self.ymin,
                    xmax=dx + self.xmax,
                    ymax=dy + self.ymax)

    def map(self, f):
        """Maps all box coordinates to a new position using a given function.
        Args:
          f: A function that takes a single coordinate and returns a new one.
        Returns:
          A :obj:`BBox` with the new coordinates.
        """
        return BBox(xmin=f(self.xmin),
                    ymin=f(self.ymin),
                    xmax=f(self.xmax),
                    ymax=f(self.ymax))

    @staticmethod
    def intersect(a, b):
        """Gets a box representing the intersection between two boxes.
        Args:
          a: :obj:`BBox` A.
          b: :obj:`BBox` B.
        Returns:
          A :obj:`BBox` representing the area where the two boxes intersect
          (may be an invalid box, check with :func:`valid`).
        """
        return BBox(xmin=max(a.xmin, b.xmin),
                    ymin=max(a.ymin, b.ymin),
                    xmax=min(a.xmax, b.xmax),
                    ymax=min(a.ymax, b.ymax))

    @staticmethod
    def union(a, b):
        """Gets a box representing the union of two boxes.
        Args:
          a: :obj:`BBox` A.
          b: :obj:`BBox` B.
        Returns:
          A :obj:`BBox` representing the unified area of the two boxes
          (always a valid box).
        """
        return BBox(xmin=min(a.xmin, b.xmin),
                    ymin=min(a.ymin, b.ymin),
                    xmax=max(a.xmax, b.xmax),
                    ymax=max(a.ymax, b.ymax))

    @staticmethod
    def iou(a, b):
        """Gets the intersection-over-union value for two boxes.
        Args:
          a: :obj:`BBox` A.
          b: :obj:`BBox` B.
        Returns:
          The intersection-over-union value: 1.0 meaning the two boxes are
          perfectly aligned, 0 if not overlapping at all (invalid intersection).
        """
        intersection = BBox.intersect(a, b)
        if not intersection.valid:
            return 0.0
        area = intersection.area
        return area / (a.area + b.area - area)


class DetectionRawOutput(collections.namedtuple('DetectionRawOutput', ['boxes', 'class_ids', 'scores', 'count'])):
    """Represents the raw output tensors of the interpreter.
        .. py:attribute:: boxes
            Array containing raw values for all boxes that outcome from the detection
        .. py:attribute:: class_ids
            Array containing raw values for all class ids that outcome from the detection
        .. py:attribute:: scores
            Array containing raw values for all scores that outcome from the detection
        .. py:attribute:: count
            Integer value representing the number of objects that outcome from the detection
    """
    __slots__ = ()

    def save_to_file(self, filename):
        common.save_tensors_to_file(self._asdict(), filename)

    @staticmethod
    def from_data(data):
        return DetectionRawOutput(
            boxes=data['boxes'],
            class_ids=data['class_ids'],
            scores=data['scores'],
            count=data['count'])

    @staticmethod
    def from_file(filename):
        data = common.load_tensors_from_file(filename)
        return DetectionRawOutput.from_data(data)

    def objs_from_data(data, threshold=-float('inf')):
        det_out = data.get('detection_output')
        img_scale = data.get('input_image_scale')
        input_size = data.get('model_input_size')
        if type(det_out) is np.ndarray:
            objs = [Object.from_nparray(obj_data, input_size, img_scale) for obj_data in det_out]
            return list(filter(lambda o: o.score >= threshold, objs))
        elif type(det_out) is dict:
            det_raw_out = DetectionRawOutput.from_data(det_out)
            return det_raw_out.get_objects(input_size, img_scale, threshold)

    @staticmethod
    def objs_from_file(filename, threshold=-float('inf')):
        data = common.load_tensors_from_file(filename)
        return DetectionRawOutput.objs_from_data(data, threshold)

    def get_objects(self, input_size, img_scale=(1., 1.), threshold=-float('inf'), nobjs=None, nparray=False):
        count = nobjs if not nobjs is None else self.count
        width, height = input_size
        img_scale_x, img_scale_y = img_scale
        sx, sy = width / img_scale_x, height / img_scale_y

        def make_object(i):
            if nparray: 
                return np.concatenate(([int(self.class_ids[i]), self.scores[i]], self.boxes[i]))
            else:
                ymin, xmin, ymax, xmax = self.boxes[i]
                return Object(
                    id=int(self.class_ids[i]),
                    score=self.scores[i],
                    bbox=BBox(xmin, ymin, xmax, ymax).scale(sx, sy).map(int))

        objs = [make_object(i) for i in range(count) if self.scores[i] >= threshold]
        return np.array(objs, dtype=np.float32) if nparray else objs


def get_detection_raw_output(interpreter):
    return DetectionRawOutput(
        boxes=common.output_tensor(interpreter, 0)[0],
        class_ids=common.output_tensor(interpreter, 1)[0],
        scores=common.output_tensor(interpreter, 2)[0],
        count=int(common.output_tensor(interpreter, 3)[0]))


def get_objects(interpreter, img_scale=(1., 1.), threshold=-float('inf'), nobjs=None, nparray=False):
    input_size = common.input_size(interpreter)
    return get_detection_raw_output(interpreter).get_objects(input_size, img_scale, threshold, nobjs, nparray)

def draw_detections_and_show(model_name,img_name, detections, labels, color='green'):
    image = Image.open(common.get_image_file_from_name(model_name,img_name)).convert('RGB')
    draw = ImageDraw.Draw(image)
    for obj in detections:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                    outline=color)
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                fill=color)
    image.save(model_name+"_"+img_name+"_golden.jpg")

def match_detections(gold_dets: List[Object], sdc_dets: List[Object]):
    n_gold = len(gold_dets)
    n_sdc = len(sdc_dets)

    iou_matrix = np.zeros((n_gold, n_sdc))
    for i, gold_det in enumerate(gold_dets):
        for j, sdc_det in enumerate(sdc_dets):
            iou = BBox.iou(gold_det.bbox, sdc_det.bbox)
            iou_matrix[i, j] = iou
            # Best IoU, so the other bboxes can be ignored
            if iou == 1.0: break

    matched_dets: List[Tuple[Object, Object, int]] = []

    if n_gold > n_sdc:
        for sdc_idx in range(n_sdc):
            best_iou_gold_idx = np.argmax(iou_matrix[:, sdc_idx])
            best_iou = iou_matrix[best_iou_gold_idx, sdc_idx]
            matched_dets.append((gold_dets[best_iou_gold_idx], sdc_dets[sdc_idx], best_iou))
    else:
        for gold_idx in range(n_gold):
            best_iou_sdc_idx = np.argmax(iou_matrix[gold_idx, :])
            best_iou = iou_matrix[gold_idx, best_iou_sdc_idx]
            matched_dets.append((gold_dets[gold_idx], sdc_dets[best_iou_sdc_idx], best_iou))

    return matched_dets

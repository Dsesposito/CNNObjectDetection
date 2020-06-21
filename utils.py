import collections
from collections import Iterable

import cv2
import numpy
from object_detection.utils.visualization_utils import draw_bounding_box_on_image_array, \
    STANDARD_COLORS


class DataLabel(object):
    def __init__(self):
        self.data = None
        self.labels = None


def draw_boxes_on_image(
        image, boxes, classes, scores, category_index, min_scores_thresh,
        line_thickness=4):

    if not isinstance(min_scores_thresh, Iterable):
        min_scores_thresh_per_class = {
            class_data['name']: min_scores_thresh
            for class_data in category_index.values()
        }
    elif isinstance(min_scores_thresh, dict):
        min_scores_thresh_per_class = {
            class_data['name']: min_scores_thresh['others']
            for class_data in category_index.values()
        }
        min_scores_thresh.pop('others')
        min_scores_thresh_per_class.update(min_scores_thresh)
    else:
        min_scores_thresh_per_class = {
            class_data['name']: 0.5
            for class_data in category_index.values()
        }

    drawn_boxes = []
    for i in range(boxes.shape[0]):
        class_name = category_index[classes[i]]['name']

        if scores[i] < min_scores_thresh_per_class[class_name]:
            continue

        drawn_boxes.append((class_name, boxes[i]))

    for class_name, box in drawn_boxes:
        height, width = image.shape[:2]
        ymin, xmin, ymax, xmax = box
        pt1 = pt1_x, pt1_y = (int(xmin*width), int(ymin*height))
        pt2 = pt2_x, pt2_y = (int(xmax*width), int(ymax*height))
        color = list(numpy.random.random(size=3) * 256)
        cv2.rectangle(image, pt1, pt2, color, thickness=2)

        font_face = cv2.FONT_HERSHEY_PLAIN
        thickness = 1
        baseline = 0
        ((text_width, text_height), _) = cv2.getTextSize(class_name, font_face, thickness, baseline)

        #cv2.rectangle(image, pt1, (pt1_x + text_width, pt1_y + text_height), (255, 255, 255), -1)
        cv2.putText(
            image, class_name, (pt1_x + 2*thickness, pt1_y + text_height + 2*thickness),
            fontFace=font_face, fontScale=1, color=(0, 0, 0), thickness=thickness
        )

    return image

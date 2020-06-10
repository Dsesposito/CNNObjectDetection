import collections
from collections import Iterable

from object_detection.utils.visualization_utils import draw_bounding_box_on_image_array, \
    STANDARD_COLORS


class DataLabel(object):
    def __init__(self):
        self.data = None
        self.labels = None


def visualize_boxes_and_labels_on_image_array(
        image, boxes, classes, scores, category_index, min_scores_thresh,
        line_thickness=4):

    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)

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

    for i in range(boxes.shape[0]):
        class_name = category_index[classes[i]]['name']

        if scores[i] < min_scores_thresh_per_class[class_name]:
            continue

        box = tuple(boxes[i].tolist())

        display_str = str(class_name)
        box_to_display_str_map[box].append(display_str)

        box_to_color_map[box] = STANDARD_COLORS[
            classes[i] % len(STANDARD_COLORS)]

    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        draw_bounding_box_on_image_array(
            image, ymin, xmin, ymax, xmax, color=color, thickness=line_thickness,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=True
        )

    return image

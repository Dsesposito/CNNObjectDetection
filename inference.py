import csv
import os
import random
import string
from enum import Enum

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from object_detection.utils import label_map_util, visualization_utils

import utils

class InferenceCase(Enum):
    DATASET_INFERENCE = 'DATASET_INFERENCE'
    PEOPLE_UNDER_RAIN = 'PEOPLE_UNDER_RAIN'
    HOUSE_TREE_PEOPLE = 'HOUSE_TREE_PEOPLE'


class ObjectDetectionInference(object):
    def __init__(self):
        self.data_path = os.path.join(os.getcwd(), 'data')
        self.source_images_path = os.path.join(self.data_path, 'quick_draw_images')
        self.detected_images_path = os.path.join(self.data_path, 'detected_images')
        self.test_images_path = os.path.join(self.data_path, 'combined_drawings', 'test')
        self.projective_test_person_under_rain_images_path = os.path.join(
            self.data_path, 'projective_test_images', 'person_under_rain')
        self.projective_test_landscape_images_path = os.path.join(
            self.data_path, 'projective_test_images', 'landscape')
        self.projective_tests_detected_images_path = os.path.join(
            self.data_path, 'projective_test_images', 'detection')

        self.model_dir = os.path.join(
            os.getcwd(), 'models', 'checkpoint_model',
            'frozen_inference_graph.pb'
        )
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_dir, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.detection_graph = detection_graph

        self.classes_names = self.read_classes_names()
        labels_dir = os.path.join(
            self.data_path, 'tf_records', 'quick_draw_object_detection_train_dataset.pbtxt'
        )
        label_map = label_map_util.load_labelmap(labels_dir)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=len(self.classes_names), use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        self.category_index = category_index

    def read_classes_names(self):
        return [
            d for d in os.listdir(self.source_images_path)
            if os.path.isdir(os.path.join(self.source_images_path, d))
        ]

    def evaluate_test_images(self):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                with open(os.path.join(self.test_images_path, 'annotations.csv')) as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    random_indexes = np.random.randint(1, 10000, 100)
                    selected_rows = [
                        row for idx, row in enumerate(csv_reader) if idx in random_indexes
                    ]
                    for row in selected_rows:
                        image = np.array(
                            plt.imread(os.path.join(self.test_images_path, row['file_name']))
                        )

                        detected_image = self.detect_on_image(image, sess)

                        filename, file_extension = os.path.splitext(row['file_name'])
                        img_path = os.path.join(
                            self.detected_images_path, '{}.png'.format(filename))

                        self.draw_image_on_plot(detected_image, img_path)

    def draw_image_on_plot(self, image, output_path):
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image, aspect='auto')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(plt.gcf())

    def detect_on_image(self, image, sess, case=None):
        image_expanded = image[np.newaxis, :, :]
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        detected_image = utils.draw_boxes_on_image(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            min_scores_thresh=self.get_thresholds(case),
            line_thickness=1,
        )
        return detected_image

    def get_thresholds(self, case=None):
        if case == InferenceCase.DATASET_INFERENCE:
            return 0.5
        elif case == InferenceCase.PEOPLE_UNDER_RAIN:
            return {
                'hand': 0.3, 'sun': 0.3, 'star': 0.3, 't-shirt': 0.3, 'house': 0.8, 'barn': 0.8,
                'pants': 0.3, 'umbrella': 0.1, 'smiley face': 0.3, 'shorts': 0.3, 'face': 0.3,
                'moon': 0.3, 'leg': 0.9, 'foot': 0.6, 'cloud': 0.3, 'shoe': 0.3, 'tree': 0.8,
                'rain': 0.8, 'others': 0.8
            }
        elif case == InferenceCase.HOUSE_TREE_PEOPLE:
            return {
                'hand': 0.8, 'sun': 0.3, 'star': 0.3, 't-shirt': 0.8, 'house': 0.5, 'barn': 0.5,
                'pants': 0.8, 'umbrella': 0.8, 'smiley face': 0.8, 'shorts': 0.8, 'face': 0.8,
                'moon': 0.3, 'leg': 0.95, 'foot': 0.8, 'cloud': 0.7, 'shoe': 0.8, 'tree': 0.3,
                'rain': 0.9, 'others': 0.8
            }
        else:
            return 0.5

    def evaluate_projective_test(self, images_files_path, case):
        for image_path in images_files_path:
            image = np.array(
                plt.imread(image_path)
            )

            with self.detection_graph.as_default():
                with tf.Session(graph=self.detection_graph) as sess:
                    detected_image = self.detect_on_image(image, sess, case)

                    file_name = '{}.png'.format(
                        ''.join(random.choices(string.ascii_letters + string.digits, k=16)))

                    detected_img_path = os.path.join(
                        self.projective_tests_detected_images_path, file_name
                    )

                    self.draw_image_on_plot(detected_image, detected_img_path)

    def evaluate_person_under_rain(self):
        images_files_path = [
            os.path.join(
                self.projective_test_person_under_rain_images_path, file_name
            ) for file_name in
            os.listdir(self.projective_test_person_under_rain_images_path)
            if not os.path.isdir(
                os.path.join(self.projective_test_person_under_rain_images_path, file_name)
            )
        ]

        self.evaluate_projective_test(images_files_path, case=InferenceCase.PEOPLE_UNDER_RAIN)

    def evaluate_landscape(self):
        images_files_path = [
            os.path.join(self.projective_test_landscape_images_path, file_name) for file_name in
            os.listdir(self.projective_test_landscape_images_path)
            if not os.path.isdir(
                os.path.join(self.projective_test_landscape_images_path, file_name)
            )
        ]

        self.evaluate_projective_test(images_files_path, case=InferenceCase.HOUSE_TREE_PEOPLE)

    def evaluate_projective_tests(self):
        self.evaluate_person_under_rain()
        self.evaluate_landscape()

    def plot_train_metrics(self):
        events_path = os.path.join(os.getcwd(), 'models', 'events')
        eval_events = [
            os.path.join(events_path, 'eval', 'events.out.tfevents.1592087785.ip-172-31-10-247'),
            os.path.join(events_path, 'eval', 'events.out.tfevents.1592099936.ip-172-31-10-247'),
            os.path.join(events_path, 'eval', 'events.out.tfevents.1592107084.ip-172-31-10-247'),
            os.path.join(events_path, 'eval', 'events.out.tfevents.1592149091.ip-172-31-10-247'),
        ]
        train_events = [
            os.path.join(events_path, 'train', 'events.out.tfevents.1592085847.ip-172-31-10-247'),
            os.path.join(events_path, 'train', 'events.out.tfevents.1592097743.ip-172-31-10-247'),
            os.path.join(events_path, 'train', 'events.out.tfevents.1592104843.ip-172-31-10-247'),
            os.path.join(events_path, 'train', 'events.out.tfevents.1592146753.ip-172-31-10-247'),
        ]
        eval_tags = {
            'DetectionBoxes_Precision/mAP@.50IOU': 'map_at_50_iou',
            'DetectionBoxes_Precision/mAP@.75IOU': 'map_at_75_iou',
            'loss': 'eval_total_loss',
            'Loss/classification_loss': 'classification_loss',
            'Loss/localization_loss': 'localization_loss',
        }
        data_points = {
            'map_at_50_iou': [],
            'map_at_75_iou': [],
            'eval_total_loss': [],
            'train_total_loss': [],
            'classification_loss': [],
            'localization_loss': []
        }
        for event_file in eval_events:
            for event in tf.train.summary_iterator(event_file):
                for value in event.summary.value:
                    if value.tag in eval_tags.keys():
                        data_points[eval_tags[value.tag]].append((event.step, value.simple_value))

        train_tags = {
            'loss_1': 'train_total_loss',
        }
        for event_file in train_events:
            for event in tf.train.summary_iterator(event_file):
                for value in event.summary.value:
                    if value.tag in train_tags.keys():
                        data_points[train_tags[value.tag]].append((event.step, value.simple_value))
                    else:
                        continue

        self.plot_data(
            data_points, 'classification_loss', 'Validation Classification Loss',
            'loss', 'steps', 'eval_classification_loss.png', annotation_index=8,
            annotation_kwargs={'xytext': (0, -5)}
        )
        self.plot_data(
            data_points, 'localization_loss', 'Validation Localization Loss',
            'loss', 'steps', 'eval_localization_loss.png', annotation_index=8,
            annotation_kwargs={'xytext': (0, -5)}
        )
        self.plot_data(
            data_points, 'eval_total_loss', 'Validation Total Loss', 'loss', 'steps',
            'eval_total_loss.png', annotation_index=8, annotation_kwargs={'xytext': (0, -10)}
        )
        self.plot_data(
            data_points, 'train_total_loss', 'Training Total Loss', 'loss', 'steps',
            'train_total_loss.png', left_x_lim=2500, top_y_lim=10, annotation_index=297,
            annotation_kwargs={'xytext': (0, 35)}
        )
        self.plot_data(
            data_points, 'map_at_50_iou', 'Mean Average Precision at 50% IoU', 'MAP @ 50 IoU',
            'steps', 'map_at_50_iou.png', annotation_index=8, annotation_kwargs={'xytext': (0, 5)}
        )
        self.plot_data(
            data_points, 'map_at_75_iou', 'Mean Average Precision at 75% IoU', 'MAP @ 75 IoU',
            'steps', 'map_at_75_iou.png', annotation_index=8, annotation_kwargs={'xytext': (0, 5)}
        )

    def plot_data(self, eval_data_points, metric, title, ylabel, xlabel, file_name, left_x_lim=None,
                  top_y_lim=None, close_fig=True, annotation_index=None, annotation_kwargs=None):
        events_path = os.path.join(os.getcwd(), 'models', 'events')
        steps, losses = map(list, zip(*eval_data_points[metric]))
        plt.plot(steps, losses, marker='.')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if left_x_lim:
            plt.xlim(left=left_x_lim)
        if top_y_lim:
            plt.ylim(top=top_y_lim)
        plt.grid(True)

        if annotation_index:
            if not annotation_kwargs:
                annotation_kwargs = {'xytext': (0, 0)}
            ax = plt.gca()
            ax.annotate(
                s='( {} , {:.2f} )'.format(steps[annotation_index], losses[annotation_index]),
                xy=(steps[annotation_index], losses[annotation_index]),
                horizontalalignment='center',
                verticalalignment='center',
                textcoords='offset pixels',
                **annotation_kwargs
            )

        if close_fig:
            plt.savefig(os.path.join(events_path, file_name))
            plt.close(plt.gcf())


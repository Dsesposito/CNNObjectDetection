import csv
import itertools
import json
import random
import string

import h5py
import os
from os.path import  join

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from object_detection.utils import dataset_util

from utils import DataLabel


class ObjectDetectionDataSetHandler(object):
    def __init__(self):
        self.train = DataLabel()
        self.test = DataLabel()
        self.validation = DataLabel()
        self.canvas_size = (400, 400, 4)
        self.grid_size = (2, 2)
        self.classes_name = []

        self.column_names = ['size_x', 'size_y', 'class_name_index', 'class_name', 'corner_y',
                             'corner_x', 'corner_y2', 'corner_x2']
        formats = [np.uint8, np.uint8, np.uint8, np.dtype('U32'), np.uint8, np.uint8, np.uint8,
                   np.uint8]
        self.labels_type = dict(names=self.column_names, formats=formats)

        self.data_path = join(os.getcwd(), 'data')
        self.combined_drawings_path = join(self.data_path, 'combined_drawings')
        self.source_images_path = join(self.data_path, 'quick_draw_images')

        self.tf_records_path = join(self.data_path, 'tf_records')

        self.read_classes_names()

    def read_classes_names(self):
        self.classes_name = [
            d for d in os.listdir(self.source_images_path)
            if os.path.isdir(os.path.join(self.source_images_path, d))
        ]

    def combine_drawings(self):
        classes_path = [
            os.path.join(self.source_images_path, d) for d in os.listdir(self.source_images_path)
            if os.path.isdir(os.path.join(self.source_images_path, d))
        ]
        self.classes_name = [
            d for d in os.listdir(self.source_images_path)
            if os.path.isdir(os.path.join(self.source_images_path, d))
        ]
        inverted_classes = {
            class_name: index for index, class_name in enumerate(self.classes_name)
        }
        amt_classes = len(classes_path)
        images_amt_per_canvas = 4
        images_per_class = 12000

        images_files_path = np.zeros((images_per_class, amt_classes), dtype=np.dtype('U512'))
        for i, class_path in enumerate(classes_path):
            images_files_path[:, i] = [
                os.path.join(class_path, file_name) for file_name in os.listdir(class_path)
            ][:images_per_class]

        random_indexes = np.zeros(
            (images_per_class*amt_classes, 2), dtype=np.uint64)
        for i in range(amt_classes):
            random_indexes[
                i*images_per_class:(i+1)*images_per_class, :
            ] = list(
                zip(range(images_per_class), i*np.ones((images_per_class, ), dtype=np.uint64))
            )

        np.random.shuffle(random_indexes)

        amt_combined_images = images_per_class * amt_classes // images_amt_per_canvas
        grouped_random_indexes = random_indexes.reshape(
            (amt_combined_images, images_amt_per_canvas, 2))
        train_amt_images = int(np.floor(0.9 * amt_combined_images))

        combined_images_path = np.zeros(
            (amt_combined_images, images_amt_per_canvas), dtype=np.dtype('U512'))
        for i in range(amt_combined_images):
            for j in range(images_amt_per_canvas):
                image_index, class_index = grouped_random_indexes[i, j, :]
                combined_images_path[i, j] = images_files_path[image_index, class_index]

        source_images_size = 100
        channels_amt = 4  # RGBA
        train_annotations = []
        test_annotations = []
        for index, combined_image_path in enumerate(combined_images_path):
            images_group = np.zeros(
                (images_amt_per_canvas, source_images_size, source_images_size, channels_amt),
                dtype=np.float32
            )
            labels_indexes = [None] * images_amt_per_canvas
            for i, image_path in enumerate(combined_image_path):
                images_group[i, :, :, :] = plt.imread(image_path)
                decomposed_path = image_path.split(os.sep)
                labels_indexes[i] = inverted_classes[decomposed_path[-2]]

            canvas, annotation = self._combine_drawings_group(images_group, labels_indexes)

            file_name = '{}.jpg'.format(
                ''.join(random.choices(string.ascii_letters + string.digits, k=16)))
            if index < train_amt_images:
                path = os.path.join(self.combined_drawings_path, 'train', file_name)
                train_annotations.append((annotation, file_name))
            else:
                path = os.path.join(self.combined_drawings_path, 'test', file_name)
                test_annotations.append((annotation, file_name))

            plt.imsave(path, canvas, format="jpg", vmin=0, vmax=1)

            print('Image {} of {} was generated'.format(index, amt_combined_images))

        with open(os.path.join(
                self.combined_drawings_path, 'train', 'annotations.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['images_annotation', 'file_name'])
            for annotation, file_name in train_annotations:
                writer.writerow([json.dumps(annotation), file_name])

        with open(os.path.join(
                self.combined_drawings_path, 'test', 'annotations.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['images_annotation', 'file_name'])
            for annotation, file_name in test_annotations:
                writer.writerow([json.dumps(annotation), file_name])

    def _combine_drawings_group(self, group, labels):
        canvas = np.zeros(self.canvas_size, dtype=np.float32)

        canvas_grid = list(itertools.product(range(self.grid_size[0]), range(self.grid_size[1])))
        grid_w, grid_h = ((self.canvas_size[0]) // self.grid_size[0],
                          (self.canvas_size[1]) // self.grid_size[1])

        amt_drawings_per_group = group.shape[0]
        drawing_width = group.shape[1]
        drawing_height = group.shape[2]
        grid_locations_indexes = np.random.choice(range(len(canvas_grid)), amt_drawings_per_group,
                                                  replace=False)

        annotation = [] # np.zeros((amt_drawings_per_group,), dtype=self.labels_type)
        for i, drawing in enumerate(group):
            image_size_x, image_size_y, _ = drawing.shape

            grid_locations = canvas_grid[grid_locations_indexes[i]]
            random_displacement = np.random.randint(0, (grid_w - drawing_width))
            random_corner_x = grid_locations[0] * grid_w + random_displacement
            random_displacement = np.random.randint(0, (grid_h - drawing_height))
            random_corner_y = grid_locations[1] * grid_h + random_displacement

            pt1_y, pt1_x = random_corner_y, random_corner_x
            pt2_y, pt2_x = random_corner_y + image_size_y, random_corner_x + image_size_x
            canvas[pt1_y: pt2_y, pt1_x: pt2_x, :] += drawing

            annotation.append(dict(zip(self.column_names, (
                image_size_x, image_size_y, labels[i], self.classes_name[labels[i]], pt1_y, pt1_x,
                pt2_y, pt2_x
            ))))

        return canvas, annotation

    def create_tf_records(self):
        source_path = os.path.join(self.combined_drawings_path, 'train')
        output_path = os.path.join(
            self.tf_records_path, 'quick_draw_object_detection_train_dataset.record')
        self.create_tf_record(source_path, output_path)

        source_path = os.path.join(self.combined_drawings_path, 'test')
        output_path = os.path.join(
            self.tf_records_path, 'quick_draw_object_detection_test_dataset.record')
        self.create_tf_record(source_path, output_path)

    def create_tf_record(self, source_path, output_path):
        pbtxt_path = output_path.replace('.record', '.pbtxt')
        with open(pbtxt_path, 'w') as file:
            for index, class_name in enumerate(self.classes_name):
                data = ('item {{\n\tid: {class_id}\n\tname: '
                        '"{class_name}"\n}}\n').format(class_id=index + 1, class_name=class_name)
                file.write(data)

        writer = tf.io.TFRecordWriter(output_path)

        with open(os.path.join(source_path, 'annotations.csv')) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                image = plt.imread(os.path.join(source_path, row['file_name']))
                image_annotations = json.loads(row['images_annotation'])

                tf_example = self.create_tf_example(image, image_annotations, row['file_name'])
                writer.write(tf_example.SerializeToString())

        writer.close()

    def create_tf_example(self, image, annotations, file_name):
        encoded_png = cv2.imencode('.png', image)
        encoded_png = encoded_png[1]
        encoded_png = encoded_png.tobytes()

        width, height, _ = image.shape
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        file_name, _ = os.path.splitext(file_name)
        file_name = str(file_name).encode('utf8')

        for annotation in annotations:
            xmins.append(annotation['corner_x'] / width)
            xmaxs.append(annotation['corner_x2'] / width)
            ymins.append(annotation['corner_y'] / height)
            ymaxs.append(annotation['corner_y2'] / height)
            classes_text.append(annotation['class_name'].encode('utf8'))
            classes.append(annotation['class_name_index'] + 1)

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(file_name),
            'image/source_id': dataset_util.bytes_feature(file_name),
            'image/encoded': dataset_util.bytes_feature(encoded_png),
            'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example

    # def save_samples(self):
    #     base_path = os.path.join(os.getcwd(), 'data', 'sample_images')
    #     maximum = 10
    #     if len(os.listdir(base_path)) >= maximum and not self.force_update:
    #         return
    #
    #     for index, image in enumerate(self.train.data):
    #         image_copy = np.copy(image)
    #         if index >= maximum:
    #             break
    #
    #         annotations = self.train.labels[index]
    #         for annotation in annotations:
    #             _, _, label_index, pt1_y, pt1_x, pt2_y, pt2_x = annotation
    #             pt1 = pt1_x, pt1_y
    #             pt2 = pt2_x, pt2_y
    #             cv2.rectangle(image_copy, pt1, pt2, (255, 255, 255), thickness=1)
    #             cv2.putText(image_copy, str(self.classes_name[label_index]), (pt1_x, pt1_y - 3),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255))
    #
    #         plt.imshow(image_copy, cmap='gray')
    #         plt.savefig(os.path.join(base_path, '{}.png'.format(index)))
    #         plt.close(plt.gcf())











    # def build_data_set(self, train, test, validation, classes_names):
    #     if self.loaded and not self.force_update:
    #         return
    #
    #     self._build_object_detection_data_set(train, self.train)
    #     self._build_object_detection_data_set(test, self.test)
    #     self._build_object_detection_data_set(validation, self.validation)
    #     self.classes_name = classes_names
    #
    #     self.save_dataset()
    #
    # def _build_object_detection_data_set(self, raw_data_set, combined_data_set):
    #     total_amt_images = len(raw_data_set.data)
    #     images_amt_per_canvas = 4
    #     amt_divisible = (total_amt_images // images_amt_per_canvas) * images_amt_per_canvas
    #
    #     raw_data_set.data = raw_data_set.data * 255
    #     raw_data_set.data = raw_data_set.data.astype(np.uint8)
    #     raw_data_set.data = raw_data_set.data[:amt_divisible, :, :]
    #     raw_data_set.data = raw_data_set.data.reshape((total_amt_images // images_amt_per_canvas,
    #                                                    images_amt_per_canvas,
    #                                                    raw_data_set.data.shape[1],
    #                                                    raw_data_set.data.shape[2]))
    #
    #     raw_data_set.labels = np.argmax(raw_data_set.labels, axis=1).astype(np.uint8)
    #     raw_data_set.labels = raw_data_set.labels[:amt_divisible]
    #     raw_data_set.labels = raw_data_set.labels.reshape(
    #         (total_amt_images // images_amt_per_canvas,
    #          images_amt_per_canvas))
    #
    #     self._combine_drawings(raw_data_set, combined_data_set)

    # def _combine_drawings(self, raw_data_set, combined_data_set):
    #     combined_data_set.data = np.zeros(
    #         (raw_data_set.data.shape[0], self.canvas_size[0], self.canvas_size[1]),
    #         dtype=np.uint8)
    #
    #     combined_data_set.labels = np.zeros(
    #         (raw_data_set.labels.shape[0], raw_data_set.labels.shape[1]),
    #         dtype=self.labels_type)
    #     for index, (drawings_group, labels_group) in enumerate(
    #             zip(raw_data_set.data, raw_data_set.labels)):
    #         canvas, annotation = self._combine_drawings_group(drawings_group, labels_group)
    #         combined_data_set.data[index, :, :] = canvas
    #         combined_data_set.labels[index, :] = annotation
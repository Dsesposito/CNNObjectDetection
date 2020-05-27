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
        min_images_amt_per_canvas = 4
        max_images_amt_per_canvas = 9
        images_per_class = 12000
        total_amt_images = images_per_class * amt_classes
        amt_augmentation = 5
        total_with_augmented = total_amt_images * amt_augmentation

        images_files_path = np.zeros((images_per_class, amt_classes), dtype=np.dtype('U512'))
        for i, class_path in enumerate(classes_path):
            images_files_path[:, i] = [
                os.path.join(class_path, file_name) for file_name in os.listdir(class_path)
            ][:images_per_class]

        random_indexes = np.zeros((total_with_augmented, 2), dtype=np.uint64)
        for j in range(amt_augmentation):
            for i in range(amt_classes):
                start_index = (i*images_per_class)+(j*total_amt_images)
                end_index = ((i+1)*images_per_class)+(j*total_amt_images)
                random_indexes[start_index: end_index, :
                ] = list(zip(
                        range(images_per_class),
                        i*np.ones((images_per_class, ), dtype=np.uint64)
                    ))

        np.random.shuffle(random_indexes)

        combined_images_path = []
        amt_took_images = 0
        while amt_took_images < total_with_augmented:
            group_amt = np.random.randint(min_images_amt_per_canvas, max_images_amt_per_canvas)
            group = random_indexes[
                amt_took_images:amt_took_images+group_amt, :
            ]

            group_paths = []
            for image_index, class_index in group:
                group_paths.append(
                    images_files_path[image_index, class_index]
                )

            combined_images_path.append(group_paths)
            amt_took_images += group_amt

        train_annotations = []
        test_annotations = []
        total_combined_images = len(combined_images_path)
        train_amt_images = int(np.floor(0.9 * total_combined_images))
        for index, combined_image_path in enumerate(combined_images_path):

            images_group = []
            labels_indexes = [None] * len(combined_image_path)
            for i, image_path in enumerate(combined_image_path):
                image = plt.imread(image_path)
                augmented_image = self.augment_image(image)
                images_group.append(augmented_image)
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

            print('Image {} of {} was generated'.format(index, total_combined_images))

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
        canvas = np.ones(self.canvas_size, dtype=np.float32)
        canvas[:, :, 3] = np.zeros(self.canvas_size[0:2])

        canvas_width = self.canvas_size[0]
        canvas_height = self.canvas_size[1]

        annotation = []
        for i, drawing in enumerate(group):
            image_size_x, image_size_y, _ = drawing.shape

            random_corner_x = np.random.randint(canvas_width - image_size_x)
            random_corner_y = np.random.randint(canvas_height - image_size_y)

            pt1_y, pt1_x = random_corner_y, random_corner_x
            pt2_y, pt2_x = random_corner_y + image_size_y, random_corner_x + image_size_x
            canvas[pt1_y: pt2_y, pt1_x: pt2_x, 3] += drawing[:, :, 3]

            annotation.append(dict(zip(self.column_names, (
                image_size_x, image_size_y, labels[i], self.classes_name[labels[i]], pt1_y, pt1_x,
                pt2_y, pt2_x
            ))))

        canvas[canvas > 1] = 1
        selection = canvas[:, :, 3] > 0
        selection = selection[:, :, np.newaxis]
        selection = np.repeat(selection, 4, 2)
        selection[:, :, 3] = False
        canvas[selection] = 0
        return canvas, annotation

    def augment_image(self, image):
        random_scale = np.random.uniform(low=0.35, high=0.65)
        height, width = image.shape[:2]

        re_height = int(height * random_scale)
        re_width = int(width * random_scale)

        dim = (re_width, re_height)
        augmented_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        # augmented_image[augmented_image < 0] = 0
        # augmented_image[augmented_image > 1] = 1

        canvas = np.ones(image.shape, dtype=np.float32)
        canvas[:, :, 3] = np.zeros(image.shape[0:2])

        start_row = width // 2 - re_width // 2
        end_row = start_row + re_width
        start_column = height // 2 - re_height // 2
        end_column = start_column + re_height
        canvas[start_row:end_row, start_column:end_column, :] = augmented_image

        return canvas

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

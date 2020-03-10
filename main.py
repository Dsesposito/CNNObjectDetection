import itertools
import h5py
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import png
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from object_detection.utils import dataset_util


class DataLabel(object):
    def __init__(self):
        self.data = None
        self.labels = None


class DataSetHandler(object):
    def __init__(self, force_update=False):
        self.train = DataLabel()
        self.test = DataLabel()
        self.val = DataLabel()
        self.amt_classes = None
        self.classes_name = None
        self.loaded = False
        self.force_update = force_update

        path = join(os.getcwd(), 'data')
        file_path = join(path, 'quick_draw_dataset.hdf5')
        if os.path.exists(file_path) and not self.force_update:
            self.load_data_set()
            self.loaded = True

    def load_data_set(self):
        path = join(os.getcwd(), 'data', 'quick_draw_dataset.hdf5')
        f = h5py.File(path, 'r')
        self.train.data = np.array(f['train/data'])
        self.train.labels = np.array(f['train/labels'])
        self.test.data = np.array(f['test/data'])
        self.test.labels = np.array(f['test/labels'])
        self.val.data = np.array(f['val/data'])
        self.val.labels = np.array(f['val/labels'])
        self.amt_classes = f['meta'].attrs['amt_classes']
        self.classes_name = f['meta'].attrs['classes_name']

    def save_dataset_as_images(self):
        path = join(os.getcwd(), 'data')
        train_path = join(path, 'train')
        test_path = join(path, 'test')
        validation_path = join(path, 'validation')
        if (os.path.exists(train_path) and os.path.exists(test_path) and
                os.path.exists(validation_path) and not self.force_update):
            return

        if not os.path.exists(join(path, 'train')):
            os.mkdir(join(path, 'train'))
        if not os.path.exists(join(path, 'test')):
            os.mkdir(join(path, 'test'))
        if not os.path.exists(join(path, 'validation')):
            os.mkdir(join(path, 'validation'))

        for sub_data_set, sub_data_set_type in zip([self.train, self.test, self.val],
                                                   ['train', 'test', 'validation']):
            sub_data_set_path = join(path, sub_data_set_type)

            for (index, (image, categorical_label)) in enumerate(
                    zip(sub_data_set.data, sub_data_set.labels)):
                if index >= 0.1 * len(sub_data_set.data):
                    break
                label = int(np.argmax(categorical_label, axis=None, out=None))
                class_name = self.classes_name[label]
                image = image * 255
                image = image.reshape(image.shape[0], image.shape[1])
                image = image.astype(np.uint8)
                class_path = join(sub_data_set_path, class_name)
                if not os.path.exists(class_path):
                    os.mkdir(class_path)

                num_files = len(os.listdir(class_path))
                png_obj = png.from_array(image, mode='L')
                png_obj.save(join(class_path, '{}.png'.format(num_files)))

    def build_data_set(self):
        if self.loaded:
            return

        path = join(os.getcwd(), 'data')
        files = [file for file in listdir(path) if
                 isfile(join(path, file)) and file != 'quick_draw_data.hdf5'
                 and file != '.gitignore']
        amt_classes = len(files)
        amt_train_images = 10000
        amt_test_images = 1000
        amt_val_images = 100
        tot_amt_train_images = amt_classes * amt_train_images
        tot_amt_test_images = amt_classes * amt_test_images
        tot_amt_val_images = amt_classes * amt_val_images
        image_with_height = 28
        train_images = np.zeros((tot_amt_train_images, image_with_height, image_with_height, 1))
        train_labels = np.zeros((tot_amt_train_images,))
        test_images = np.zeros((tot_amt_test_images, image_with_height, image_with_height, 1))
        test_labels = np.zeros((tot_amt_test_images,))
        val_images = np.zeros((tot_amt_val_images, image_with_height, image_with_height, 1))
        val_labels = np.zeros((tot_amt_val_images,))
        classes_name = []
        for index, file in enumerate(files):
            file_name = os.path.splitext(file)[0]
            class_name = file_name.split('_')[-1]
            classes_name.append(class_name)
            class_images = np.load(join(path, file), allow_pickle=True)
            class_images = class_images.astype(np.float32) / 255
            np.random.shuffle(class_images)
            class_images = class_images.reshape(
                (len(class_images), image_with_height, image_with_height, 1))
            start, end = index * amt_train_images, (index + 1) * amt_train_images
            train_images[start:end] = class_images[0:amt_train_images, :, :, :]
            train_labels[start:end] = index

            start, end = index * amt_test_images, (index + 1) * amt_test_images
            start_images, end_images = amt_train_images, (amt_train_images + amt_test_images)
            test_images[start:end] = class_images[start_images:end_images, :, :, :]
            test_labels[index * amt_test_images:(index + 1) * amt_test_images] = index

            start, end = index * amt_val_images, (index + 1) * amt_val_images
            start_images = (amt_train_images + amt_test_images)
            end_images = (amt_train_images + amt_test_images + amt_val_images)
            val_images[start:end] = class_images[start_images:end_images, :, :, :]
            val_labels[index * amt_val_images:(index + 1) * amt_val_images] = index

        rand_indexes = np.random.permutation(len(train_images))
        train_images = train_images[rand_indexes, :, :, :]
        train_labels = train_labels[rand_indexes]
        train_labels = np.eye(amt_classes)[train_labels.astype(np.uint8)]

        rand_indexes = np.random.permutation(len(test_images))
        test_images = test_images[rand_indexes, :, :, :]
        test_labels = test_labels[rand_indexes]
        test_labels = np.eye(amt_classes)[test_labels.astype(np.uint8)]

        rand_indexes = np.random.permutation(len(val_images))
        val_images = val_images[rand_indexes, :, :, :]
        val_labels = val_labels[rand_indexes]
        val_labels = np.eye(amt_classes)[val_labels.astype(np.uint8)]

        self.train.data = train_images
        self.train.labels = train_labels
        self.test.data = test_images
        self.test.labels = test_labels
        self.val.data = val_images
        self.val.labels = val_labels
        self.amt_classes = amt_classes
        self.classes_name = classes_name

        self.save_data_set()

    def save_data_set(self):
        path = join(os.getcwd(), 'data')
        file_path = join(path, 'quick_draw_dataset.hdf5')

        f = h5py.File(file_path, 'w')
        f.create_dataset('train/data', data=self.train.data)
        f.create_dataset('train/labels', data=self.train.labels)
        f.create_dataset('test/data', data=self.test.data)
        f.create_dataset('test/labels', data=self.test.labels)
        f.create_dataset('val/data', data=self.val.data)
        f.create_dataset('val/labels', data=self.val.labels)
        meta_dataset = f.create_dataset('meta', shape=())
        meta_dataset.attrs['amt_classes'] = self.amt_classes
        meta_dataset.attrs['classes_name'] = self.classes_name


class ObjectDetectionDataSetHandler(object):
    def __init__(self, force_update=False):
        self.train = DataLabel()
        self.test = DataLabel()
        self.validation = DataLabel()
        self.canvas_size = (100, 100)
        self.grid_size = (2, 2)
        self.classes_name = []
        self.loaded = False
        self.force_update = force_update

        names = ['size_x', 'size_y', 'class_name_index', 'corner_y', 'corner_x', 'corner_y2',
                 'corner_x2']
        formats = [np.uint8, np.uint8, np.uint8, np.uint8, np.uint8, np.uint8, np.uint8]
        self.labels_type = dict(names=names, formats=formats)

        path = join(os.getcwd(), 'data')
        file_path = join(path, 'quick_draw_object_detection_dataset.hdf5')
        if os.path.exists(file_path):
            self.load_data_set()
            self.loaded = True

    def build_data_set(self, train, test, validation, classes_names):
        if self.loaded and not self.force_update:
            return

        self._build_object_detection_data_set(train, self.train)
        self._build_object_detection_data_set(test, self.test)
        self._build_object_detection_data_set(validation, self.validation)
        self.classes_name = classes_names

        self.save_dataset()

    def _build_object_detection_data_set(self, raw_data_set, combined_data_set):
        total_amt_images = len(raw_data_set.data)
        images_amt_per_canvas = 4
        amt_divisible = (total_amt_images // images_amt_per_canvas) * images_amt_per_canvas

        raw_data_set.data = raw_data_set.data * 255
        raw_data_set.data = raw_data_set.data.astype(np.uint8)
        raw_data_set.data = raw_data_set.data[:amt_divisible, :, :]
        raw_data_set.data = raw_data_set.data.reshape((total_amt_images // images_amt_per_canvas,
                                                       images_amt_per_canvas,
                                                       raw_data_set.data.shape[1],
                                                       raw_data_set.data.shape[2]))

        raw_data_set.labels = np.argmax(raw_data_set.labels, axis=1).astype(np.uint8)
        raw_data_set.labels = raw_data_set.labels[:amt_divisible]
        raw_data_set.labels = raw_data_set.labels.reshape(
            (total_amt_images // images_amt_per_canvas,
             images_amt_per_canvas))

        self._combine_drawings(raw_data_set, combined_data_set)

    def _combine_drawings(self, raw_data_set, combined_data_set):
        combined_data_set.data = np.zeros(
            (raw_data_set.data.shape[0], self.canvas_size[0], self.canvas_size[1]),
            dtype=np.uint8)

        combined_data_set.labels = np.zeros(
            (raw_data_set.labels.shape[0], raw_data_set.labels.shape[1]),
            dtype=self.labels_type)
        for index, (drawings_group, labels_group) in enumerate(
                zip(raw_data_set.data, raw_data_set.labels)):
            canvas, annotation = self._combine_drawings_group(drawings_group, labels_group)
            combined_data_set.data[index, :, :] = canvas
            combined_data_set.labels[index, :] = annotation

    def _combine_drawings_group(self, group, labels):
        canvas = np.zeros(self.canvas_size, dtype=np.uint8)

        canvas_grid = list(itertools.product(range(self.grid_size[0]), range(self.grid_size[1])))
        grid_w, grid_h = ((self.canvas_size[0]) // self.grid_size[0],
                          (self.canvas_size[1]) // self.grid_size[1])

        amt_drawings_per_group = group.shape[0]
        drawing_width = group.shape[1]
        drawing_height = group.shape[2]
        grid_locations_indexes = np.random.choice(range(len(canvas_grid)), amt_drawings_per_group,
                                                  replace=False)

        annotation = np.zeros((amt_drawings_per_group,), dtype=self.labels_type)
        for i, drawing in enumerate(group):
            image_size_x, image_size_y = drawing.shape

            grid_locations = canvas_grid[grid_locations_indexes[i]]
            random_displacement = np.random.randint(0, (grid_w - drawing_width))
            random_corner_x = grid_locations[0] * grid_w + random_displacement
            random_displacement = np.random.randint(0, (grid_h - drawing_height))
            random_corner_y = grid_locations[1] * grid_h + random_displacement

            pt1_y, pt1_x = random_corner_y, random_corner_x
            pt2_y, pt2_x = random_corner_y + image_size_y, random_corner_x + image_size_x
            canvas[pt1_y: pt2_y, pt1_x: pt2_x] += drawing

            annotation[i] = (image_size_x, image_size_y, labels[i], pt1_y, pt1_x,pt2_y, pt2_x)

        return canvas, annotation

    def save_dataset(self):
        path = join(os.getcwd(), 'data')
        file_path = join(path, 'quick_draw_object_detection_dataset.hdf5')
        f = h5py.File(file_path, 'w')
        f.create_dataset('train/data', data=self.train.data)
        f.create_dataset('train/labels', data=self.train.labels)
        f.create_dataset('test/data', data=self.test.data)
        f.create_dataset('test/labels', data=self.test.labels)
        f.create_dataset('val/data', data=self.validation.data)
        f.create_dataset('val/labels', data=self.validation.labels)
        meta_dataset = f.create_dataset('meta', shape=())
        meta_dataset.attrs['classes_name'] = self.classes_name

    def load_data_set(self):
        path = join(os.getcwd(), 'data', 'quick_draw_object_detection_dataset.hdf5')
        f = h5py.File(path, 'r')
        self.train.data = np.array(f['train/data'])
        self.train.labels = np.array(f['train/labels'])
        self.test.data = np.array(f['test/data'])
        self.test.labels = np.array(f['test/labels'])
        self.validation.data = np.array(f['val/data'])
        self.validation.labels = np.array(f['val/labels'])
        self.classes_name = f['meta'].attrs['classes_name']

    def create_tf_records(self):
        self.create_tf_record(self.train, 'quick_draw_object_detection_train_dataset')
        self.create_tf_record(self.test, 'quick_draw_object_detection_test_dataset')
        self.create_tf_record(self.validation, 'quick_draw_object_detection_val_dataset')

    def create_tf_record(self, dataset, file_name):
        path = join(os.getcwd(), 'data', file_name + '.record')
        if os.path.exists(path) and not self.force_update:
            return

        writer = tf.io.TFRecordWriter(path)
        for index, (image, annotation) in enumerate(zip(dataset.data, dataset.labels)):
            tf_example = self.create_tf_example(image, annotation, index)
            writer.write(tf_example.SerializeToString())
        writer.close()

        path = join(os.getcwd(), 'data', file_name + '.pbtxt')
        with open(path, 'w') as file:
            for index, class_name in enumerate(self.classes_name):
                data = ('item {{\n\tid: {class_id}\n\tname: '
                        '"{class_name}"\n}}\n').format(class_id=index + 1, class_name=class_name)
                file.write(data)

    def create_tf_example(self, image, annotations, file_name):
        encoded_png = cv2.imencode('.png', image)
        encoded_png = encoded_png[1]
        encoded_png = encoded_png.tobytes()

        width, height = image.shape
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        file_name = str(file_name).encode('utf8')

        for annotation in annotations:
            xmins.append(annotation['corner_x'] / width)
            xmaxs.append(annotation['corner_x2'] / width)
            ymins.append(annotation['corner_y'] / height)
            ymaxs.append(annotation['corner_y2'] / height)
            classes_text.append(self.classes_name[annotation['class_name_index']].encode('utf8'))
            classes.append(annotation['class_name_index'])

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

    def save_samples(self):
        base_path = os.path.join(os.getcwd(), 'data', 'images')
        maximum = 10
        if len(os.listdir(base_path)) >= maximum and not self.force_update:
            return

        for index, image in enumerate(self.train.data):
            image_copy = np.copy(image)
            if index >= maximum:
                break

            annotations = self.train.labels[index]
            for annotation in annotations:
                _, _, label_index, pt1_y, pt1_x, pt2_y, pt2_x = annotation
                pt1 = pt1_x, pt1_y
                pt2 = pt2_x, pt2_y
                cv2.rectangle(image_copy, pt1, pt2, (255, 255, 255), thickness=1)
                cv2.putText(image_copy, str(self.classes_name[label_index]), (pt1_x, pt1_y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255))

            plt.imshow(image_copy, cmap='gray_r')
            plt.savefig(os.path.join(os.getcwd(), 'images', '{}.png'.format(index)))
            plt.close(plt.gcf())


data_set = DataSetHandler(force_update=True)
data_set.build_data_set()
data_set.save_dataset_as_images()

object_detection_data_set = ObjectDetectionDataSetHandler(force_update=True)
object_detection_data_set.build_data_set(
    data_set.train, data_set.test, data_set.val, data_set.classes_name)
object_detection_data_set.save_samples()
object_detection_data_set.create_tf_records()

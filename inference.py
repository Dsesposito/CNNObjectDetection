import csv
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from object_detection.utils import label_map_util, visualization_utils


class ObjectDetectionInference(object):
    def __init__(self):
        self.data_path = os.path.join(os.getcwd(), 'data')
        self.source_images_path = os.path.join(self.data_path, 'quick_draw_images')
        self.detected_images_path = os.path.join(self.data_path, 'detected_images')
        self.test_images_path = os.path.join(self.data_path, 'combined_drawings', 'test')
        self.real_images_path = os.path.join(self.data_path, 'real_images')

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

    def process_images(self):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                with open(os.path.join(self.test_images_path, 'annotations.csv')) as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    index = 0
                    for row in csv_reader:
                        if index >= 100:
                            break

                        image = np.array(
                            plt.imread(os.path.join(self.test_images_path, row['file_name']))
                        )

                        detected_image = self.detect_on_image(image, sess)

                        img_path = os.path.join(self.detected_images_path, '{}d.png'.format(index))
                        plt.imshow(detected_image)
                        plt.savefig(img_path)
                        plt.close(plt.gcf())
                        index += 1

    def detect_on_image(self, image, sess):
        image_expanded = image[np.newaxis, :, :]
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        detected_image = visualization_utils.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=1,
            skip_scores=True,
            skip_track_ids=True,
            min_score_thresh=0.2
        )
        return detected_image

    def process_real_images(self):
        image_name = '2'
        img_path = os.path.join(self.real_images_path, '{}.jpg'.format(image_name))
        image = cv2.imread(img_path)

        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                detected_image = self.detect_on_image(image, sess)

                img_path = os.path.join(self.real_images_path, '{}d.png'.format(image_name))
                plt.imshow(detected_image)
                plt.savefig(img_path)
                plt.close(plt.gcf())
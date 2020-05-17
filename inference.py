import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from object_detection.utils import label_map_util, visualization_utils


class ObjectDetectionInference(object):
    def __init__(self, num_classes):
        model_dir = os.path.join(
            os.getcwd(), 'models', 'checkpoint_model_input_type_image_tensor',
            'frozen_inference_graph.pb'
        )
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_dir, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.detection_graph = detection_graph

        labels_dir = os.path.join(
            os.getcwd(), 'data', 'quick_draw_object_detection_test_dataset.pbtxt')
        label_map = label_map_util.load_labelmap(labels_dir)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=num_classes, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        self.category_index = category_index

    def process_images(self, images, output_dir):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                images = images[:100, :, :]
                for index, image in enumerate(images):
                    image_expanded = np.repeat(
                        image[np.newaxis, :, :, np.newaxis], 3, axis=3
                    )
                    image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                    boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                    scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_expanded})
                    image_expanded = np.repeat(
                        image[:, :, np.newaxis], 3, axis=2
                    )
                    detected_image = visualization_utils.visualize_boxes_and_labels_on_image_array(
                        image_expanded,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        self.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=1,
                        skip_scores=True,
                        skip_track_ids=True,
                        min_score_thresh=0.3
                    )
                    img_path = os.path.join(output_dir, '{}d.png'.format(index))
                    plt.imshow(detected_image)
                    plt.savefig(img_path)
                    plt.close(plt.gcf())

    def process_real_images(self):
        image_name = '5'
        img_path = os.path.join(
            os.getcwd(), 'data', 'real_images', '{}.jpg'.format(image_name))
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.bitwise_not(image)

        output_dir = os.path.join(os.getcwd(), 'data', 'real_images')
        self.process_images(image[np.newaxis, :, :], output_dir)
import os
import matplotlib.pyplot as plt

from inference import ObjectDetectionInference
from objectdetectiondatasethandler import ObjectDetectionDataSetHandler
from quickdrawdatasethandler import DataSetHandler

#data_set = DataSetHandler()
#data_set.build_data_set()

#object_detection_data_set = ObjectDetectionDataSetHandler()
#object_detection_data_set.combine_example()
#object_detection_data_set.combine_drawings()
#object_detection_data_set.create_tf_records()

inference_handler = ObjectDetectionInference()
inference_handler.evaluate_test_images()
#inference_handler.process_real_images()
#inference_handler.plot_train_metrics()

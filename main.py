from inference import ObjectDetectionInference
from objectdetectiondatasethandler import ObjectDetectionDataSetHandler
from quickdrawdatasethandler import DataSetHandler

# data_set = DataSetHandler()
# data_set.build_data_set()

object_detection_data_set = ObjectDetectionDataSetHandler()
#object_detection_data_set.combine_drawings()
object_detection_data_set.create_tf_records()

#object_detection_data_set.build_data_set(
#    data_set.train, data_set.test, data_set.val, data_set.classes_name)
#

#inference_handler = ObjectDetectionInference(len(object_detection_data_set.classes_name))
#output_path = os.path.join(os.getcwd(), 'data', 'detected_images')
#inference_handler.process_images(object_detection_data_set.test.data, output_path)
#inference_handler.process_real_images()


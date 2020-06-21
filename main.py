import click

from inference import ObjectDetectionInference
from objectdetectiondatasethandler import ObjectDetectionDataSetHandler
from quickdrawdatasethandler import DataSetHandler


@click.group()
def cli():
    pass


@cli.command()
def build_dataset():
    data_set = DataSetHandler()
    data_set.build_data_set()
    pass


@cli.command()
def build_object_detection_dataset():
    object_detection_data_set = ObjectDetectionDataSetHandler()
    object_detection_data_set.combine_drawings()


@cli.command()
def build_object_detection_images_example():
    object_detection_data_set = ObjectDetectionDataSetHandler()
    object_detection_data_set.combine_example()


@cli.command()
def build_tf_records():
    object_detection_data_set = ObjectDetectionDataSetHandler()
    object_detection_data_set.create_tf_records()


@cli.command()
def evaluate_test_images():
    inference_handler = ObjectDetectionInference()
    inference_handler.evaluate_test_images()


@cli.command()
def plot_train_metrics():
    inference_handler = ObjectDetectionInference()
    inference_handler.plot_train_metrics()


@cli.command()
def evaluate_projective_tests():
    inference_handler = ObjectDetectionInference()
    inference_handler.evaluate_projective_tests()


if __name__ == '__main__':
    cli()

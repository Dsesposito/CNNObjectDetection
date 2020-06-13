import os
import matplotlib.pyplot as plt

from inference import ObjectDetectionInference
from objectdetectiondatasethandler import ObjectDetectionDataSetHandler
from quickdrawdatasethandler import DataSetHandler

#data_set = DataSetHandler()
#data_set.build_data_set()

object_detection_data_set = ObjectDetectionDataSetHandler()
#object_detection_data_set.combine_example()
#object_detection_data_set.combine_drawings()
object_detection_data_set.create_tf_records()

#inference_handler = ObjectDetectionInference()
#inference_handler.process_images()
#inference_handler.process_real_images()

def plot_train_metrics():
    events_path = os.path.join(os.getcwd(), 'models', 'events')
    eval_events = [
        os.path.join(events_path, 'eval', 'events.out.tfevents.1590625336.ip-172-31-10-247'),
        os.path.join(events_path, 'eval', 'events.out.tfevents.1590671649.ip-172-31-10-247'),
        os.path.join(events_path, 'eval', 'events.out.tfevents.1590700385.ip-172-31-10-247'),
        os.path.join(events_path, 'eval', 'events.out.tfevents.1590709554.ip-172-31-10-247'),
    ]
    train_events = [
        os.path.join(events_path, 'train', 'events.out.tfevents.1590623453.ip-172-31-10-247'),
        os.path.join(events_path, 'train', 'events.out.tfevents.1590669525.ip-172-31-10-247'),
        os.path.join(events_path, 'train', 'events.out.tfevents.1590698179.ip-172-31-10-247'),
        os.path.join(events_path, 'train', 'events.out.tfevents.1590707291.ip-172-31-10-247'),
    ]
    import tensorflow as tf
    data_points = {
        'map_at_50_iou': [],
        'map_at_75_iou': [],
        'eval_loss': [],
        'train_loss': [],
    }
    for event_file in eval_events:
        for event in tf.train.summary_iterator(event_file):
            for value in event.summary.value:
                if value.tag == 'DetectionBoxes_Precision/mAP@.50IOU':
                    data_points['map_at_50_iou'].append((event.step, value.simple_value))
                elif value.tag == 'DetectionBoxes_Precision/mAP@.75IOU':
                    data_points['map_at_75_iou'].append((event.step, value.simple_value))
                elif value.tag == 'loss':
                    data_points['eval_loss'].append((event.step, value.simple_value))
                else:
                    continue

    for event_file in train_events:
        for event in tf.train.summary_iterator(event_file):
            for value in event.summary.value:
                if value.tag == 'loss_1':
                    data_points['train_loss'].append((event.step, value.simple_value))
                else:
                    continue

    plot_data(data_points, 'eval_loss', 'Validation Loss', 'loss', 'steps', 'eval_loss.png')
    plot_data(
        data_points, 'train_loss', 'Training Loss', 'loss', 'steps', 'train_loss.png',
        left_x_lim=2500, top_y_lim=10
    )
    plot_data(
        data_points, 'map_at_50_iou', 'Mean average precision at 50% IoU', 'MAP @ 50 IoU',
        'steps', 'map_at_50_iou.png'
    )
    plot_data(
        data_points, 'map_at_75_iou', 'Mean average precision at 75% IoU', 'MAP @ 75 IoU',
        'steps', 'map_at_75_iou.png'
    )

    plot_data(data_points, 'eval_loss', '', '', '', '', close_fig=False, left_x_lim=2500,
              top_y_lim=10)
    plot_data(
        data_points, 'train_loss', 'Training and Validation Loss', 'loss', 'steps',
        'train_and_val_loss.png', left_x_lim=2500, top_y_lim=10
    )


def plot_data(eval_data_points, metric, title, ylabel, xlabel, file_name, left_x_lim=None,
              top_y_lim=None, close_fig=True):
    events_path = os.path.join(os.getcwd(), 'models', 'events')
    steps, losses = map(list, zip(*eval_data_points[metric]))
    plt.plot(steps, losses)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if left_x_lim:
        plt.xlim(left=left_x_lim)
    if top_y_lim:
        plt.ylim(top=top_y_lim)
    plt.grid(True)
    if close_fig:
        plt.savefig(os.path.join(events_path, file_name))
        plt.close(plt.gcf())


#plot_train_metrics()

# Object Detection - Quick Draw Data set

Quick draw data set: https://github.com/googlecreativelab/quickdraw-dataset

## Generate record files

 * Run `main.py`
 * Check tf records with: `python tfviewer.py ~/Repositorios/personales/CNNObjectDetection/data/tf_records/quick_draw_object_detection_train_dataset.record --labels-to-highlight='hand;sun;star;t-shirt;house;barn;pants;umbrella;smiley face;shorts;face;moon;leg;foot;cloud;shoe;tree;rain'`

## Object Detection TensorFlow API:

https://github.com/tensorflow/models/tree/master/research/object_detection

## Install

### Local

 * Install tensorflow with pip. WARNING: INSTALL Latest Tensorflow v1
 * Clone models into tensorflow inside env directory: git clone https://github.com/tensorflow/models
 * Checkout to latest v1 with `git checkout tags/v1.13.0 -b release/v1.13.0`
 * Follow instructions from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
 * Test if installation is ok:
    * `export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`
    * Move to tensor flow dir: `cd /home/dsesposito/Repositorios/personales/CNNObjectDetection/venv/lib/python3.7/site-packages/tensorflow/models/research/`
    * `python object_detection/builders/model_builder_test.py`

### AWS

To run on AWS deploy a EC2 instance with the Deep Learning AMI. Then use the right env: `source activate tensorflow_p36`

## Train

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md

### Local
 * Activate env: `source activate ./venv/bin/activate`
 * Move to tensor flow dir: `cd /home/dsesposito/Repositorios/personales/CNNObjectDetection/venv/lib/python3.7/site-packages/tensorflow/models/research/`
 * Add slim directory to python path: `export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`
 * Execute the train script:

```
export PIPELINE_CONFIG_PATH=/home/dsesposito/Repositorios/personales/CNNObjectDetection/models/ssd_inception_v2_coco/ssd_inception_v2_coco_local.config MODEL_DIR=/home/dsesposito/Repositorios/personales/CNNObjectDetection/models/model NUM_TRAIN_STEPS=50000 SAMPLE_1_OF_N_EVAL_EXAMPLES=1
```

```
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
```

### AWS

 * Copy dataset and pre-trained-model
``` 
scp -r -i ~/Trabajo/Optiwe/platform.pem /home/dsesposito/Repositorios/personales/CNNObjectDetection/resources/tf_records.zip  ubuntu@54.153.2.153:~/object_detection/CNNObjectDetection/data/
```
 * unzip with `unzip tf_records.zip `

```
scp -r -i ~/Trabajo/Optiwe/platform.pem /home/dsesposito/Repositorios/personales/CNNObjectDetection/resources/ssd_inception_v2_coco_2018_01_28.tar.gz  ubuntu@AWS_DNS:~/object_detection/CNNObjectDetection/models/ssd_inception_v2_coco
```

 * To unzip the .tar file: `tar -xvf ssd_inception_v2_coco_2018_01_28.tar.gz`

 * Activate env: `source activate tensorflow_p36`
 * Delete previous model with `rm ~/object_detection/CNNObjectDetection/models/model/* -r`
 * Move to tensor flow dir: `cd ~/anaconda3/envs/tensorflow_p36/lib/python3.6/site-packages/tensorflow/models/research/`
 * Add slim directory to python path: `export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`
 * Execute the train script:

```
export PIPELINE_CONFIG_PATH=/home/ubuntu/object_detection/CNNObjectDetection/models/ssd_inception_v2_coco/ssd_inception_v2_coco.config MODEL_DIR=/home/ubuntu/object_detection/CNNObjectDetection/models/model NUM_TRAIN_STEPS=50000 SAMPLE_1_OF_N_EVAL_EXAMPLES=1
```
```
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
```

  * Run tensorboard: `tensorboard --logdir=${MODEL_DIR}`

## Inference

### Export model on aws

 * Activate venv `source activate tensorflow_p36`
 * Delete previous checkpoint model with `rm ~/object_detection/CNNObjectDetection/models/checkpoint_model/ -r`
 * Move to tensor flow dir: `cd ~/anaconda3/envs/tensorflow_p36/lib/python3.6/site-packages/tensorflow/models/research/`
 * Add slim directory to python path: `export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`
 * Export checkpoint (use image_tensor or tf_example as INPUT_TYPE):

```
export INPUT_TYPE=image_tensor PIPELINE_CONFIG_PATH=/home/ubuntu/object_detection/CNNObjectDetection/models/ssd_inception_v2_coco/ssd_inception_v2_coco.config TRAINED_CKPT_PREFIX=/home/ubuntu/object_detection/CNNObjectDetection/models/model/model.ckpt-17500 EXPORT_DIR=/home/ubuntu/object_detection/CNNObjectDetection/models/checkpoint_model
```

```
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
```

```
zip -r checkpoint_model.zip checkpoint_model/
```

### Download model

```
scp -i ~/Trabajo/Optiwe/platform.pem ubuntu@AWS_IP:/home/ubuntu/object_detection/CNNObjectDetection/models/checkpoint_model.zip ~/Downloads
```

### Inference on test dataset locally
 * Activate env: `source activate ./venv/bin/activate`
 * Move to tensor flow dir: `cd /home/dsesposito/Repositorios/personales/CNNObjectDetection/venv/lib/python3.7/site-packages/tensorflow/models/research/`
 * Add slim directory to python path: `export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`
 * Execute the train script:

```
export TF_RECORD_OUTPUT_FILE=/home/dsesposito/Repositorios/personales/CNNObjectDetection/data/inference_on_test_ds.tfrecord TF_RECORD_INPUT_FILES=/home/dsesposito/Repositorios/personales/CNNObjectDetection/data/quick_draw_object_detection_test_dataset.record INFERENCE_GRAPH_PATH=/home/dsesposito/Repositorios/personales/CNNObjectDetection/models/checkpoint_model_input_type_tf_example/frozen_inference_graph.pb
```
```
python -m object_detection.inference.infer_detections \
  --input_tfrecord_paths=$TF_RECORD_INPUT_FILES \
  --output_tfrecord_path=$TF_RECORD_OUTPUT_FILE \
  --inference_graph=$INFERENCE_GRAPH_PATH
```


## Resources

https://medium.com/@teyou21/setup-tensorflow-for-object-detection-on-ubuntu-16-04-e2485b52e32a
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/
https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e


### Bug fixing

If evaluation took more than 10 min. Read: https://stackoverflow.com/questions/52800897/tensorflow-runs-running-per-image-evaluation-indefinitly

```
I figured out a solution for the problem. The problem with tensorflow 1.10 and after is, that you can not set checkpoint steps or checkpoint secs in the config file like before. By default tensorflow 1.10 and after saves a checkpoint every 10 min. If your hardware is not fast enough and you need more then 10 min for evaluation, you are stuck in a loop.

So to change the time steps or training steps till a new checkpoint is safed (which triggers the evaluation), you have to navigate to the model_main.py in the following folder:

tensorflow/models/research/object_detection/

Once you opened model_main.py, navigate to line 62. Here you will find

config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir)

To trigger the checkpoint save after 2500 steps for example, change the entry to this:

config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir,save_checkpoints_steps=2500).

Now the model is saved every 2500 steps and afterwards an evaluation is done.

There are multiple parameters you can pass through this option. You can find a documentation here:

tensorflow/tensorflow/contrib/learn/python/learn/estimators/run_config.py.

From Line 231 to 294 you can see the parameters and documentation.

I hope I can help you with this and you don't have to look for an answer as long as I did.
```
# Details

This repo contains the code used to create an object detector which could been used to create an automatic psychology projective tests.

## Directories structure

Create the following directories structure

 data
 ----> combined_drawings
       ----> test
       ----> train 
 ----> detected_images
 ----> examples_quickdraw
 ----> ndjson
 ----> projective_test_images
       ----> detection
       ----> landscape
       ----> person_under_rain
 ----> quick_draw_images
 ----> tf_records

## Available commands:

 * Create base dataset: `python -m main build-dataset`
 * Create object detection dataset: `python -m main build-dataset`
 * Create object detection dataset examples: `python -m main build-object-detection-images-example`
 * Create tf records: `python -m main build-tf-records`
 * Evaluate model on test images: `python -m main evaluate-test-images`
 * Plot train metrics: `python -m main plot-train-metrics`
 * Evaluate projective tests: `python -m main evaluate-projective-tests`
 
import os
import sys
import random
import math
import numpy as np
import skimage.io
import logging

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn import visualize

# Root directory of the project
ROOT_DIR = os.path.abspath(".")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class InstanceSegmentationConfig(Config):
    NAME = "coco"
    IMAGES_PER_CPU = 1
    IMAGES_PER_GPU = 1
    CPU_COUNT = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80


def init_model():
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
       utils.download_trained_weights(COCO_MODEL_PATH)

    config = InstanceSegmentationConfig()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']
    return model, class_names


def run_detection(image, model, class_names):
    # Run detection
    results = model.detect([image], verbose=1)

    objects = []
    # Visualize results
    r = results[0]

    for offset, size_percentage, class_id in zip(r['offset'], r['size_percentage'], r['class_ids']):
        label = class_names[class_id]
        object = {"class_name": label, "offset": offset, "size_percentage": size_percentage}
        objects.append(object)

    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])

    return objects

if __name__ == "__main__":
    try:
        model, class_names = init_model()
        image = skimage.io.imread(os.path.join(IMAGE_DIR, "sample.jpg"))
        logging.info(run_detection(image, model, class_names))
    except Exception as e:
        logging.error(e)
        sys.exit()




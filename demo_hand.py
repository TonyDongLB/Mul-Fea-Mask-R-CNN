import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize

import torch
import hand


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
HAND_MODEL_PATH = '/home/dl/Work/pytorch-mask-rcnn/logs/hand20190304T1855/mask_rcnn_hand_0316.pth'

# Directory of images to run detection on
IMAGE_DIR = '/home/dl/Work/pytorch-mask-rcnn/hand_instance/images/val'

class InferenceConfig(hand.HandConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object.
model = modellib.MaskRCNN_Hand(model_dir=MODEL_DIR, config=config)
if config.GPU_COUNT:
    model = model.cuda()

# Load weights trained on Hand
model.load_state_dict(torch.load(HAND_MODEL_PATH))


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'hand']

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]

for file_name in file_names:
    image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))
    # Run detection
    results = model.detect([image])

# Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'], file_name=file_name)
    plt.show()
import os
import random
random.seed(42)
import numpy as np
np.random.seed(42)

from mask_rcnn.util.config import Config
from mask_rcnn.util import utils
from mask_rcnn.model import model as modellib
from mask_rcnn.util.cityscapes_dataset import CityscapesDataset
from mask_rcnn.model.data_generator import load_image_gt

# Root directory of the project
ROOT_DIR = '/output/cityscapes/root_dir'
os.makedirs(ROOT_DIR, exist_ok=True)

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class CityscapesConfig(Config):
    """Configuration for training on the cityscapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cityscapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 8 + 1  # background + actual classes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    # TODO
    # IMAGE_MIN_DIM = 1024
    # IMAGE_MAX_DIM = 1024
    # TODO
    IMAGE_MIN_DIM = 969
    IMAGE_MAX_DIM = 1280

    MEAN_PIXEL = np.array((72.78044, 83.21195, 73.45286))

    # Use full epoche for each dataset
    STEPS_PER_EPOCH = None

    # Use full data for validation
    VALIDATION_STEPS = None

    LEARNING_RATE = 0.002

    LEARNING_RMS_PROP_EPSILON = 1e-8
    LEARNING_ADAM_EPSILON = 1e-8
    LEARNING_ADAM_USE_AMSGRAD = True

    MAX_METRICS_IMAGES = 500

    NUM_WORKERS = 14


optimizer = 'sgd'


config = CityscapesConfig()
config.display()

cityscapes_cache_path = '/data/mask-rcnn/my_cityscapes/mask_cache'
cityscapes_cache_version = 2

# Training dataset
dataset_train = CityscapesDataset(cache_path=cityscapes_cache_path, version=cityscapes_cache_version,
                                  cache_images=False, grayscale=True)
dataset_train.load_images('train')
dataset_train.prepare()

# Validation dataset
dataset_val = CityscapesDataset(cache_path=cityscapes_cache_path, version=cityscapes_cache_version,
                                cache_images=False, grayscale=True)
dataset_val.load_images('val')
dataset_val.prepare()


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)



# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
num_epochs = 5  # TODO
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=num_epochs,
            layers='heads', optimizer_type=optimizer)

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
num_epochs = 10  # TODO
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=num_epochs,
            layers="all", optimizer_type=optimizer)

num_epochs = 15  # TODO
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=num_epochs,
            layers="all", optimizer_type=optimizer)



# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "mask_rcnn_cityscapes_eval.h5")
model.keras_model.save_weights(model_path)


class InferenceConfig(CityscapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()


# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        load_image_gt(dataset_val, inference_config,
                      image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    if len(gt_class_id) != 0:
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id,
                             r["rois"], r["class_ids"], r["scores"])
        APs.append(AP)

print("mAP: ", np.mean(APs))
import argparse
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')

from mask_rcnn.util.config import Config
from mask_rcnn.util.inference_dataset import InferenceDataset
from mask_rcnn.util.plot_datasets import plot_datasets


def main(dataset_name):
    print(f'Plotting dataset {dataset_name}.')
    sys.stdout.flush()

    output_dir = f'/output/{dataset_name}'
    image_folder = f'/data/{dataset_name}'

    class CityscapesConfig(Config):
        """Configuration for training on the cityscapes dataset.
        Derives from the base Config class and overrides values specific
        to the toy shapes dataset.
        """
        # Give the configuration a recognizable name
        NAME = dataset_name

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

    config = CityscapesConfig()

    inference_cache_path = f'/output/{dataset_name}/mask_cache'
    inference_cache_version = 2
    # Training dataset
    dataset = InferenceDataset(dataset_name=dataset_name, image_folder=image_folder,
                               cache_path=inference_cache_path, version=inference_cache_version,
                               cache_images=False, grayscale=True)
    dataset.load_images()
    dataset.prepare()

    class InferenceConfig(CityscapesConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()

    plot_datasets(output_dir=output_dir, config=config, inference_config=inference_config,
                  datasets=[dataset])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer and plot bounding boxes.')
    parser.add_argument('dataset_name', type=str,
                        help='the name of the dataset, e.g. cityscapes if you want to process /data/cityscapes')

    args = parser.parse_args()
    main(dataset_name=args.dataset_name)

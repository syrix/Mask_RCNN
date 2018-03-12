import os
import sys

from tqdm import tqdm

from mask_rcnn.util import utils
from mask_rcnn.model import model as modellib
from mask_rcnn.util import visualize
from mask_rcnn.model.data_generator import load_image_gt


def _infere_and_plot(dataset, plot_dir, model, inference_config):
    image_ids = dataset.image_ids
    for image_id in tqdm(image_ids):
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            load_image_gt(dataset, inference_config, image_id, use_mini_mask=False)

        filename = os.path.basename(dataset.source_image_link(image_id))
        filename_without_extension = os.path.splitext(filename)[0]

        save_path = os.path.join(plot_dir, f'{filename_without_extension}_gt.png')
        visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                    dataset.class_names, save_path=save_path)

        results = model.detect([original_image])

        r = results[0]

        save_path = os.path.join(plot_dir, f'{filename_without_extension}_pred.png')
        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                    dataset.class_names, r['scores'], save_path=save_path)


def plot_datasets(output_dir, config, inference_config, datasets, model_path=None):
    # Root directory of the project
    root_dir = os.path.join(output_dir, 'root_dir')
    os.makedirs(root_dir, exist_ok=True)
    # Directory to save logs and trained model
    model_dir = os.path.join(root_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    # Local path to trained weights file
    coco_model_path = os.path.join(root_dir, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(coco_model_path):
        utils.download_trained_weights(coco_model_path)

    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    config.display()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=model_dir)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(root_dir, ".h5 file name here")
    if model_path is None:
        model_path = model.find_last()[1]

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print(f'Loading weights from {model_path}')
    model.load_weights(model_path, by_name=True)

    # Plot validation images
    for dataset in datasets:
        print(f'INFO: Plot {len(dataset.image_ids)} images')
        sys.stdout.flush()
        _infere_and_plot(dataset=dataset, plot_dir=plot_dir, model=model, inference_config=inference_config)

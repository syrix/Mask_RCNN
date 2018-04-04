import os
from collections import namedtuple
from collections import defaultdict

from skimage import color
import skimage.io
import numpy as np

from mask_rcnn.util.dataset import CachedDataset


# a label and all meta information
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'train_id',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'train_name',

    'category',  # The name of the category that this label belongs to

    'category_id',  # The ID of this category. Used to create ground truth images
    # on category level.

    'has_instances',  # Whether this label distinguishes between single instances or not

    'ignore_in_eval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])

labels = [
    # name, id, trainId, train_name, category, catId, hasInstances, ignoreInEval, color
    Label('unlabeled', 0, 1, 'void', 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 1, 'void', 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 1, 'void', 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 1, 'void', 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 1, 'void', 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 1, 'void', 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 1, 'void', 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 2, 'road', 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 3, 'sidewalk', 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 1, 'void', 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 1, 'void', 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 4, 'building', 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 5, 'wall', 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 6, 'fence', 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 1, 'void', 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 1, 'void', 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 1, 'void', 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 7, 'pole', 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 1, 'void', 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 8, 'traffic light', 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 9, 'traffic sign', 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 10, 'vegetation', 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 11, 'terrain', 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 12, 'sky', 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 13, 'person', 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 14, 'rider', 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 15, 'car', 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 16, 'truck', 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 17, 'bus', 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 1, 'void', 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 1, 'void', 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 18, 'train', 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 19, 'motorcycle', 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 20, 'bicycle', 'vehicle', 7, True, False, (119, 11, 32)),
    # TODO Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


# name to label object
name2label = {label.name: label for label in labels}
# id to label object
id2label = {label.id: label for label in labels}
# id to trainId
id2train_id = {label.id: label.train_id for label in labels}


class InferenceDataset(CachedDataset):
    def __init__(self, dataset_name, image_folder, class_map=None,
                 cache_path='', version='', cache_images=True, cache_masks=True,
                 grayscale=False):
        self.grayscale = grayscale
        self.dataset_name = dataset_name
        self.image_folder = image_folder

        super().__init__(class_map=class_map, cache_path=cache_path, version=version, cache_images=cache_images,
                         cache_masks=cache_masks)

    def load_images(self):
        """Load images.
        """

        # Add classes
        #         classes = {
        #             1: 'ego vehicle',
        #             2: 'rectification border',
        #             3: 'out of roi',
        #             4: 'static',
        #             5: 'dynamic',
        #             6: 'ground',
        #             7: 'road',
        #             8: 'sidewalk',
        #             9: 'parking',
        #             10: 'rail track',
        #             11: 'building',
        #             12: 'wall',
        #             13: 'fence',
        #             14: 'guard rail',
        #             15: 'bridge',
        #             16: 'tunnel',
        #             17: 'pole',
        #             18: 'polegroup',
        #             19: 'traffic light',
        #             20: 'traffic sign',
        #             21: 'vegetation',
        #             22: 'terrain',
        #             23: 'sky',
        #             24: 'person',
        #             25: 'rider',
        #             26: 'car',
        #             27: 'truck',
        #             28: 'bus',
        #             29: 'caravan',
        #             30: 'trailer',
        #             31: 'train',
        #             32: 'motorcycle',
        #             33: 'bicycle',
        #             # TODO -1: 'license plate',
        #         }

        #         for id, name in classes.items():
        #             self.add_class("cityscapes", id, name)

        allowed_label_names = [
            'person',
            'rider',
            'car',
            'truck',
            'bus',
            'train',
            'motorcycle',
            'bicycle',
            'traffic light',
            'traffic sign'
        ]
        allowed_labels = [label for label in labels if label.name in allowed_label_names]

        self.id2internal_id = defaultdict(lambda: -1)

        counter = 1
        for label in allowed_labels:
            if label.train_id != 0:
                self.id2internal_id[label.id] = counter
                self.add_class(self.dataset_name, label.train_id, label.train_name)
                counter += 1

        # Add images
        for count, filename in enumerate(os.listdir(self.image_folder)):
            image_id = f'img-{count}'
            image_path = os.path.join(self.image_folder, filename)
            self.add_image(self.dataset_name, image_id=image_id, path=image_path)

    def load_image_without_caching(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        image = skimage.io.imread(info['path'])

        if self.grayscale:
            image = color.rgb2gray(image)
            image = np.expand_dims(image, axis=2)
            image = np.repeat(image, repeats=3, axis=2)
            image = image * 255  # convert from float to int representation
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == self.dataset_name:
            return image_id
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask_without_caching(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


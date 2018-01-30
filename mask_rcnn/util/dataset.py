import gzip
import os

import numpy as np
from tqdm import tqdm

from mask_rcnn.util.utils import Dataset


class CachedDataset(Dataset):
    """
    Dataset base class for datasets where images, masks or both should be cached on disk.
    This will lead to a performance gain if masks are expensive to compute and the hard disk is fast.
    """
    def __init__(self, class_map=None, cache_path='', version='', cache_images=True, cache_masks=True):
        assert cache_path != '' and version != '', 'cache_path and version can not be empty'
        self.cache_path = os.path.join(cache_path, f'v_{version}')
        self.cache_images = cache_images
        self.cache_masks = cache_masks
        self.data_cached = False

        os.makedirs(self.cache_path, exist_ok=True)

        super().__init__(class_map)

    def _cache_path(self, image_id, path_type):
        name = self.image_info[image_id]['id']
        filename = f'{name}.{path_type}.npy.gz'
        path = os.path.join(self.cache_path, filename)
        return path

    def image_path(self, image_id):
        return self._cache_path(image_id, 'image')

    def masks_path(self, image_id):
        return self._cache_path(image_id, 'masks')

    def classes_path(self, image_id):
        return self._cache_path(image_id, 'classes')

    def is_cached(self, image_id):
        cached = True
        # shortcut once we know everything is cached
        if self.data_cached:
            return True
        if self.cache_images and not os.path.isfile(self.image_path(image_id)):
            cached = False
        if self.cache_masks and not os.path.isfile(self.masks_path(image_id)):
            cached = False
        if self.cache_masks and not os.path.isfile(self.classes_path(image_id)):
            cached = False
        return cached

    def prepare(self, class_map=None):
        super().prepare(class_map)

        print('Preparing dataset...')

        self.data_cached = False
        for image_id in tqdm(self.image_ids):
            # images already present
            if self.is_cached(image_id):
                continue

            if self.cache_images:
                image = self.load_image(image_id)
                with gzip.GzipFile(self.image_path(image_id), 'w', compresslevel=1) as f:
                    np.save(f, image)
            if self.cache_masks:
                masks, classes = self.load_mask(image_id)
                with gzip.GzipFile(self.masks_path(image_id), 'w', compresslevel=1) as f:
                    np.save(f, masks)
                with gzip.GzipFile(self.classes_path(image_id), 'w', compresslevel=1) as f:
                    np.save(f, classes)
        self.data_cached = True

    def load_image(self, image_id):
        if self.cache_images and self.is_cached(image_id):
            with gzip.GzipFile(self.image_path(image_id), 'r', compresslevel=1) as f:
                return np.load(f)
        return self.load_image_without_caching(image_id)

    def load_image_without_caching(self, image_id):
        raise Exception('not implemented')

    def load_mask(self, image_id):
        if self.cache_masks and self.is_cached(image_id):
            with gzip.GzipFile(self.masks_path(image_id), 'r', compresslevel=1) as f:
                masks = np.load(f)
            with gzip.GzipFile(self.classes_path(image_id), 'r', compresslevel=1) as f:
                classes = np.load(f)
            return masks, classes
        return self.load_mask_without_caching(image_id)

    def load_mask_without_caching(self, image_id):
        raise Exception('not implemented')

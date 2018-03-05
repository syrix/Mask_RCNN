import numpy as np

from mask_rcnn.util.cityscapes_dataset import _split_mask_into_instances


class TestSplitMaskIntoInstances:
    def test_it_leaves_a_connected_mask_unchanged(self):
        input_mask = np.expand_dims(np.array(
            [[0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0]]), axis=2)
        expected_output = [input_mask.copy()]

        output = _split_mask_into_instances(input_mask)

        assert np.all(np.array(output) == np.array(expected_output))

    def test_it_splits_disconnected_masks(self):
        input_mask = np.expand_dims(np.array(
            [[0, 0, 0, 0, 0],
             [0, 1, 0, 1, 0],
             [0, 1, 0, 1, 0],
             [0, 0, 0, 0, 0]]), axis=2)
        expected_mask_1 = np.expand_dims(np.array(
            [[0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0]]), axis=2)
        expected_mask_2 = np.expand_dims(np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0]]), axis=2)
        expected_output = [expected_mask_1, expected_mask_2]

        output = _split_mask_into_instances(input_mask)

        assert np.all(np.array(output) == np.array(expected_output))

    def test_it_does_not_split_masks_connected_by_diagonal_pixels(self):
        input_mask = np.expand_dims(np.array(
            [[0, 0, 1, 0, 0],
             [0, 1, 0, 1, 0],
             [0, 1, 0, 1, 0],
             [0, 0, 0, 0, 0]]), axis=2)
        expected_output = [input_mask.copy()]

        output = _split_mask_into_instances(input_mask)

        assert np.all(np.array(output) == np.array(expected_output))

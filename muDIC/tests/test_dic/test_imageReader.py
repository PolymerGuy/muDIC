from unittest import TestCase

import numpy as np

from muDIC.IO import ImageStack
from muDIC.IO.image_stack import ImageReader


def read_image_mocked(path, rotate=False):
    return np.zeros((100, 100))


def find_file_names_mocked(type=".png"):
    return ['file' + str(i).zfill(2) + type for i in range(100)]


class MockedImageStackFromFiles(ImageStack):
    def __init__(self, image_reader):
        super(MockedImageStackFromFiles, self).__init__(image_reader)

    # TODO: Use mocking instead of overwriting class methods
    def __read_image__(self, *args, **kwargs):
        return read_image_mocked(args, kwargs)

    def __find_file_names__(self, *args, **kwargs):
        return find_file_names_mocked(args, kwargs)


class MockerImageReader(ImageReader):
    def __init__(self):
        paths = find_file_names_mocked()
        super(MockerImageReader, self).__init__(paths)


class TestImageReader(TestCase):
    # Tests a an subclass of ImageReader where two methods have be overridden
    def setUp(self):
        path_to_folder = '/just/a/path/'
        image_reader = MockerImageReader()
        self.images = MockedImageStackFromFiles(image_reader)

    def test_get_active_frame_ids(self):
        # Check that the frame ids correspond to the images

        # Do several passes to check for side effects
        for i in range(5):
            # Get list of frame ids
            ids = self.images._active_img_ids_
            # Get list of frame ids from img names
            correct_images = [elm for ind, elm in enumerate(self.images.image_reader._image_paths_) if ind in ids]
            # Extract image ids from image names
            img_ids = [int((img.replace('file', '')).replace('.png', '')) for img in correct_images]

            # Check if all ids have a corresponding image
            self.assertEquals(ids, img_ids)

            # Remove some images and repeat
            self.images.skip_images([2, 7, 17, 92])

    def test_skip_images(self):
        skipped_frames = []

        # Try removing the frames individually
        for frame in range(0, 99, 3):
            skipped_frames.append(frame)
            self.images.skip_images(skipped_frames)
            ids = self.images._active_img_ids_
            img_ids = self.images._active_img_ids_  # [int((img.replace('file', '')).replace('.png', '')) for img in self.images._active_img_ids_]
            self.assertEquals(ids, img_ids)

        # Exception when index is out of bounds
        self.assertRaises(ValueError, self.images.skip_images, [101])
        self.assertRaises(ValueError, self.images.skip_images, [-1])

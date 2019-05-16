import logging
import os
from copy import copy
from functools import partial

import numpy as np
from scipy import ndimage


class ImageStack(object):
    def __init__(self, image_reader, filter=None):
        """
        Image stack

        ImageStacks are responsible for handling image imports, filtering and ordering.


        Various functionality is available, such as:

        * Adding a filter
        * Skipping images
        * Reversing the order of the images

         Parameters
         ----------
         image_reader : ImageReader or ImageGetter
             An object which handles the import of the images
         filter : Filter
             A function which filters the image

         Examples
         ----------
        >>> import muDIC as dic
        >>> import numpy as np
        >>> image_list = [np.ones((3,3))*i for i in range(10)]
        >>> image_stack = dic.image_stack_from_list(image_list)
        >>> image_stack.set_filter(dic.filtering.lowpass_gaussian,sigma=2)
        >>> image_stack.skip_images(range(9))
        >>> image_stack[0]
            array([[9., 9., 9.],
            [9., 9., 9.],
            [9., 9., 9.]])

         """

        self.logger = logging.getLogger()
        self.image_reader = image_reader
        self._all_img_ids_ = list(range(len(self.image_reader)))
        self._active_img_ids_ = copy(self._all_img_ids_)
        if filter is None:
            self._filter_ = lambda img: img
        else:
            self._filter_ = filter

    def set_filter(self, filter, **kwargs):
        """
        Set a filter

        The filter is applied to all frames.
        The filter has to be a function which takes an image as input and returns an image.

         Parameters
         ----------
         filter : func
             A function which takes an image as input and returns an image
         kwargs :
             The arguments that should be used for the filter.
         """
        self._filter_ = partial(filter, **kwargs)

    def __getitem__(self, index):
        return self._filter_(self.image_reader(self._active_img_ids_[index]))

    def revere_order(self):
        """
        Reverse the order of the images in the stack

        """
        self._active_img_ids_ = self._active_img_ids_[::-1]
        self.logger.info("Reversing the order of the image stack")

    def __len__(self):
        return len(self._active_img_ids_)

    def skip_images(self, skip_frames):
        """
        Skip the frames with the ids listed in skip_frames

         Parameters
         ----------
         skip_frames : list or tuple
             The list of frame ids to be skipped
         """

        if type(skip_frames) is not list and type(skip_frames) is not tuple:
            raise TypeError('Frames has to be given as a list or tuple with integers')

        elif not all((type(ind) == int) for ind in skip_frames):
            raise TypeError('Frame ids have to be integers')

        elif max(skip_frames) > len(self._all_img_ids_) or min(skip_frames) < 0:
            raise ValueError('Frame id outside bounds')

        self._active_img_ids_ = [frame for frame in self._active_img_ids_ if frame not in skip_frames]
        self.logger.info("Skipping frames. Length of image stack is now %i" % len(self._active_img_ids_))

    def use_every_n_image(self, n):

        self._active_img_ids_ = self._active_img_ids_[::n]
        self.logger.info("Using every %i frame. Length of image stack is now %i" % (n, len(self._active_img_ids_)))


class ImageReader(object):
    def __init__(self, image_paths):
        """
        Image reader which reads files from HD

        When the ImageReader is called, it returns an image corresponding to the index.

         Parameters
         ----------
         image_paths : list
             A list of paths to images.

         """
        self._image_paths_ = image_paths
        self.precision = np.float64

    def __len__(self):
        return len(self._image_paths_)

    def __call__(self, index, rotate=False):
        if not rotate:
            return ndimage.imread(self._image_paths_[index], flatten=True).astype(self.precision)
        else:
            return ndimage.rotate(ndimage.imread(self._image_paths_[index], flatten=True).astype(self.precision),
                                  rotate)


class ImageListWrapper(object):
    def __init__(self, images):
        """
        Image getter which wraps a list of images

        When the ImageGetter is called, it returns an image corresponding to the index.

         Parameters
         ----------
         images : list
             A list of images as numpy.ndarray

         """
        self._images_ = images
        self.precision = np.float64

    def __len__(self):
        return len(self._images_)

    def __call__(self, index, rotate_ang=False):
        if not rotate_ang:
            return self._images_[index].astype(self.precision)
        else:
            return ndimage.rotate(self._images_[index].astype(self.precision), angle=rotate_ang)


def find_file_names(path, type=".png"):
    """
    Finds all files with the given extension in the folder path.

     Parameters
     ----------
     path : str
         The path to the folder containing the files of interest
     type : str
         The file postfix such as ".png", ".bmp" etc.

     Returns
     -------
    List of filenames

     """
    return sorted([os.path.join(path, file) for file in os.listdir(path) if file.endswith(type)])


def image_stack_from_folder(path_to_folder, file_type='.png'):
    """
    Make an ImageStack containing the images within a folder


     Parameters
     ----------
     path_to_folder : str
         The path to the folder containing the images of interest
     file_type : str
         The file postfix such as ".png", ".bmp" etc.

     Returns
     -------
     ImageStack object


     Examples
     --------
    >>> path_to_folder = r"/the/path/to/the/images/"
    >>> image_stack = image_stack_from_folder(path_to_folder,file_type=".bmp")
    ImageStack
     """
    logger = logging.getLogger()
    supported_filetypes = ['.png', '.bmp', '.tif']

    if type(path_to_folder) not in [str]:
        raise TypeError('Path has to be a string')

    if type(file_type) is not str or file_type not in supported_filetypes:
        # TODO: include more valid file types
        raise TypeError('Filetype has to be: %s' % str.join(*supported_filetypes))

    file_names_all = find_file_names(path_to_folder, file_type)

    logger.info("Found %i images in folder" % len(file_names_all))

    return ImageStack(ImageReader(file_names_all))


def image_stack_from_list(image_list):
    """
    Make an ImageStack containing the images in the list


     Parameters
     ----------
     image_list : list or tuple
         The list of images as numpy.ndarray

     Returns
     -------
     ImageStack object


     Examples
     --------
        >>> import muDIC as dic
        >>> import numpy as np
        >>> image_list = [np.random.rand(10,10) for i in range(10)]
        >>> image_stack = dic.image_stack_from_folder(image_list)
    ImageStack
     """
    logger = logging.getLogger()

    if type(image_list) not in [list, tuple]:
        raise TypeError('image_list has to be either a list or tuple')

    if type(image_list[0]) is not np.ndarray:
        raise TypeError("Images has to be numpy.ndarrays")

    logger.info("Found %i images in list" % len(image_list))

    return ImageStack(ImageListWrapper(image_list))

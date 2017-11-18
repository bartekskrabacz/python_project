import os
import logging
import numpy as np

from os import listdir
from os.path import join, isdir

from sklearn.utils import Bunch
from sklearn.externals.joblib import Memory
import shutil


logger = logging.getLogger(__name__)

class Data_importer:
    def __init__(self):
        self.current_directory = os.path.dirname(os.path.abspath(__file__ + "../../"))
        print self.current_directory
        self.lfw_home = join(self.current_directory, "resources/data")
        self.data_folder_path = join(self.lfw_home, "lfw_funneled")

    def load_images(self, file_paths, slice_, color, resize):
        """Internally used to load images"""

        # Try to import imread and imresize from PIL. We do this here to prevent
        # the whole sklearn.datasets module from depending on PIL.
        try:
            try:
                from scipy.misc import imread
            except ImportError:
                from scipy.misc.pilutil import imread
            from scipy.misc import imresize
        except ImportError:
            raise ImportError("The Python Imaging Library (PIL)"
                              " is required to load data from jpeg files")

        # compute the portion of the images to load to respect the slice_ parameter
        # given by the caller
        default_slice = (slice(0, 250), slice(0, 250))
        if slice_ is None:
            slice_ = default_slice
        else:
            slice_ = tuple(s or ds for s, ds in zip(slice_, default_slice))

        h_slice, w_slice = slice_
        h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)
        w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)

        if resize is not None:
            resize = float(resize)
            h = int(resize * h)
            w = int(resize * w)

        # allocate some contiguous memory to host the decoded image slices
        n_faces = len(file_paths)
        if not color:
            faces = np.zeros((n_faces, h, w), dtype=np.float32)
        else:
            faces = np.zeros((n_faces, h, w, 3), dtype=np.float32)

        # iterate over the collected file path to load the jpeg files as numpy
        # arrays
        for i, file_path in enumerate(file_paths):
            if i % 1000 == 0:
                logger.debug("Loading face #%05d / %05d", i + 1, n_faces)

            # Checks if jpeg reading worked. Refer to issue #3594 for more
            # details.
            img = imread(file_path)
            if img.ndim is 0:
                raise RuntimeError("Failed to read the image file %s, "
                                   "Please make sure that libjpeg is installed"
                                   % file_path)

            face = np.asarray(img[slice_], dtype=np.float32)
            face /= 255.0  # scale uint8 coded colors to the [0.0, 1.0] floats
            if resize is not None:
                face = imresize(face, resize)
            if not color:
                # average the color channels to compute a gray levels
                # representation
                face = face.mean(axis=2)

            faces[i, ...] = face

        return faces

    def load_data(self, data_folder_path, slice_=None, color=False, resize=None,
                      min_faces_per_person=0):
        """Perform the actual data loading for the lfw people dataset

        This operation is meant to be cached by a joblib wrapper.
        """
        # scan the data folder content to retain people with more that
        # `min_faces_per_person` face pictures
        person_names, file_paths = [], []
        for person_name in sorted(listdir(data_folder_path)):
            folder_path = join(data_folder_path, person_name)
            if not isdir(folder_path):
                continue
            paths = [join(folder_path, f) for f in sorted(listdir(folder_path))]
            n_pictures = len(paths)
            if n_pictures >= min_faces_per_person:
                person_name = person_name.replace('_', ' ')
                person_names.extend([person_name] * n_pictures)
                file_paths.extend(paths)

        n_faces = len(file_paths)
        if n_faces == 0:
            raise ValueError("min_faces_per_person=%d is too restrictive" %
                             min_faces_per_person)

        target_names = np.unique(person_names)
        target = np.searchsorted(target_names, person_names)

        faces = self.load_images(file_paths, slice_, color, resize)

        # shuffle the faces with a deterministic RNG scheme to avoid having
        # all faces of the same person in a row, as it would break some
        # cross validation and learning algorithms such as SGD and online
        # k-means that make an IID assumption

        indices = np.arange(n_faces)
        np.random.RandomState(42).shuffle(indices)
        faces, target = faces[indices], target[indices]
        return faces, target, target_names

    def import_data(self, resize=0.5, min_faces_per_person=0, color=False,
                         slice_=(slice(70, 195), slice(78, 172))):
        """Loader for the Labeled Faces in the Wild (LFW) people dataset

        This dataset is a collection of JPEG pictures of famous people
        collected on the internet, all details are available on the
        official website:

            http://vis-www.cs.umass.edu/lfw/

        Each picture is centered on a single face. Each pixel of each channel
        (color in RGB) is encoded by a float in range 0.0 - 1.0.

        The task is called Face Recognition (or Identification): given the
        picture of a face, find the name of the person given a training set
        (gallery).

        The original images are 250 x 250 pixels, but the default slice and resize
        arguments reduce them to 62 x 47.

        Parameters
        ----------

        resize : float, optional, default 0.5
            Ratio used to resize the each face picture.

        min_faces_per_person : int, optional, default None
            The extracted dataset will only retain pictures of people that have at
            least `min_faces_per_person` different pictures.

        color : boolean, optional, default False
            Keep the 3 RGB channels instead of averaging them to a single
            gray level channel. If color is True the shape of the data has
            one more dimension than the shape with color = False.

        slice_ : optional
            Provide a custom 2D slice (height, width) to extract the
            'interesting' part of the jpeg files and avoid use statistical
            correlation from the background


        Returns
        -------
        dataset : dict-like object with the following attributes:

        dataset.data : numpy array of shape (13233, 2914)
            Each row corresponds to a ravelled face image of original size 62 x 47
            pixels. Changing the ``slice_`` or resize parameters will change the
            shape of the output.

        dataset.images : numpy array of shape (13233, 62, 47)
            Each row is a face image corresponding to one of the 5749 people in
            the dataset. Changing the ``slice_`` or resize parameters will change
            the shape of the output.

        dataset.target : numpy array of shape (13233,)
            Labels associated to each face image. Those labels range from 0-5748
            and correspond to the person IDs.

        dataset.DESCR : string
            Description of the Labeled Faces in the Wild (LFW) dataset.
        """
        logger.debug('Loading LFW people faces from %s', self.lfw_home)

        # wrap the loader in a memoizing function that will return memmaped data
        # arrays for optimal memory usage
        m = Memory(cachedir=self.lfw_home, compress=6, verbose=0)
        load_func = m.cache(self.load_data)

        # load and memoize the pairs as np arrays
        faces, target, target_names = load_func(
            self.data_folder_path, resize=resize,
            min_faces_per_person=min_faces_per_person, color=color, slice_=slice_)

        # pack the results as a Bunch instance
        return Bunch(data=faces.reshape(len(faces), -1), images=faces,
                     target=target, target_names=target_names,
                     DESCR="LFW faces dataset")

    def clear_cache(self):
        shutil.rmtree(self.lfw_home +"/joblib")
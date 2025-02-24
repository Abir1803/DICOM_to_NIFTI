import warnings
from typing import Union, Callable
import nibabel as nib
import numpy as np
import torchio as tio


def check_list_affines(list_affine: list):
    """
    Checks the list of images' affine transform, if they are multiple modalities in the dataset.
    :param list_affine: Lis t of affine transforms for all found images
    :return: bool : True if the affine transforms are all the same, else False
    """
    for i in range(len(list_affine)):
        for j in range(i + 1, len(list_affine)):
            if np.allclose(list_affine[i], list_affine[j]):  # TODO: use almost_equal function from numpy
                pass
            else:
                return False
    return True


class AffinesNotMatchingError(ValueError):
    def __init__(self, message):
        super(AffinesNotMatchingError, self).__init__(message)


class PatientNII2NPY:
    def __init__(self, paths_images: Union[list[str], str], path_labels: str, strict_affine_verification: bool = True,
                 transform: Callable = None):
        self.paths_images = paths_images
        self.path_labels = path_labels
        self.nb_images = 1
        self.transform = transform
        self.tio_subject_kwargs = {}
        if isinstance(self.paths_images, list):
            self.nb_images = len(self.paths_images)
            if self.nb_images == 1:
                self.paths_images = self.paths_images[0]
        if self.nb_images == 1:
            self.img_affine = None
            self.img_array = None
        else:
            self.img_affines = []
            self.img_arrays = []
        self.label_array = None
        self.label_affine = None
        self.label_shape = None
        self.image_shape = None
        self.strict_affine_verification = strict_affine_verification

    def check_image_shape(self):
        equal = True
        if self.nb_images == 1:
            self.image_shape = self.tio_subject_kwargs["img"][tio.DATA].shape
            self.label_shape = self.tio_subject_kwargs["mask"][tio.DATA].shape
        print(self.image_shape)
        print(self.label_shape)
        if not np.array_equal(self.image_shape, self.label_shape):
            msg = f"Found different shapes between label and image. Label shape was  {self.label_shape}," \
                  f"and image shape was {self.image_shape}"
            warnings.warn(msg)
            equal = False
        return equal, (min(self.image_shape[1], self.label_shape[1]),
                       min(self.image_shape[2], self.label_shape[2]),
                       min(self.image_shape[3], self.label_shape[3]))

    def load_image(self):
        """
        Loads image(s) and verifies their affine transforms are the same, if a list of images is given
        :return:
        """
        if self.nb_images == 1:
            img = tio.ScalarImage(self.paths_images)
            self.img_affine = img[tio.AFFINE]
            self.check_affine_compatibility()
            self.img_array = img[tio.DATA]
            self.tio_subject_kwargs["img"] = img

        else:
            for path in self.paths_images:
                img = tio.ScalarImage(path)
                if self.transform is not None:
                    img = self.transform(img)
                self.img_affines.append(img.affine)
                self.img_arrays.append(np.array(img.dataobj))
            if not check_list_affines(self.img_affines):
                msg = f"Found different affines in images : {self.img_affines}"
                if self.strict_affine_verification:
                    raise AffinesNotMatchingError(msg)
                else:
                    warnings.warn(msg)

    def load_label(self):
        label = tio.LabelMap(self.path_labels)
        if self.nb_images != 1:
            if self.transform is not None:
                label = self.transform(label)

        self.label_array = label[tio.DATA]
        self.label_affine = label[tio.AFFINE]
        self.label = label
        self.label_shape = np.shape(self.label_array.numpy())
        self.tio_subject_kwargs["mask"] = label

    def check_affine_compatibility(self):
        if self.nb_images == 1:
            img_affine = self.img_affine
        else:
            img_affine = self.img_affines[0]
        if not np.allclose(img_affine, self.label_affine):
            msg = f"Found different affines between label and image. Label affine was  {self.label_affine}," \
                  f"and image affine was {img_affine}"
            if self.strict_affine_verification:
                raise AffinesNotMatchingError(msg)
            else:
                warnings.warn(msg)

    def save(self):
        if self.nb_images == 1:
            np.save(f"{self.paths_images[:-7]}.npy", self.img_array)
            np.save(f"{self.paths_images[:-7]}_affine.npy", self.img_affine)
        else:
            for idx in range(len(self.paths_images)):
                np.save(f"{self.paths_images[idx][:-7]}.npy", self.img_arrays[idx])
                np.save(f"{self.paths_images[idx][:-7]}_affine.npy", self.img_affines[idx])
        np.save(f"{self.path_labels[:-7]}.npy", self.label_array)
        np.save(f"{self.path_labels[:-7]}_affine.npy", self.label_affine)

    def process(self):
        self.load_label()
        self.load_image()
        equal, self.final_shape = self.check_image_shape()
        sub = tio.Subject(self.tio_subject_kwargs)
        sub = self.transform(sub)
        if not equal:
            print(f"Resizing towards {self.final_shape}")
            tr_resize = tio.CropOrPad(self.final_shape)
            sub = tr_resize(sub)
        self.img_array = sub["img"][tio.AFFINE]
        self.img_affine = sub["img"][tio.AFFINE]
        self.label_array = sub["mask"][tio.AFFINE]
        self.label_affine = sub["mask"][tio.AFFINE]
        self.save()

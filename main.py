import json
import logging
import os
import pickle
import shutil
import sys
import traceback
from typing import List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm
import SimpleITK as sitk  # NOQA
import warnings
import torchio as tio

from utils.data.io import maybe_mkdir
from utils.format.nii2npy import PatientNII2NPY, AffinesNotMatchingError
from utils.data.sorter import DICOMSorter
from utils.format.DICOM2NRRD import DirectoryDICOMToNRRD
from utils.data.verification import check_dcm_series, clear_wrong_uids, check_spacing_ok
from utils.format.DICOM2NII import PatientDICOMRTToNII, DirectoryDICOMToNII


class MedDecathlonFolderBuilderFromDICOM:
    """
    This class can be used to build a "Medical Decathlon"-like dataset from a dump of DICOM files i.e. a folder in
    which they are not sorted DICOM files
    """
    def __init__(self, task_id: int, task_name: str, src_dir: str, inter_dir: str, dest_dir:str, logger:logging.Logger,  # NOQA
                 validation_split: float = 1 / 3, reference: str = "", licence: str = "MIT",
                 release: str = "1.0 1/1/1970", modalities: dict = None,
                 rois_dict: dict = None, aliases_dict: dict = None, nn_unet_version: int = 2, force_redo_copy=True):  # NOQA
        """

        :param task_id: The ID for the task, which must be between 1 and 999. For custom tasks,
        it is recommended to choose an ID above 500.
        :param task_id: The name of the task
        :param src_dir: Directory in which all the DICOM files are stored.
        DICOM files need to be at the root of the folder, not in sub-folders. #TODO add recursive reading of DICOM files
        :param dest_dir: Directory where the Medical Decathlon style dataset will be created
        :param logger: Logger to log errors and warnings
        :param validation_split: the part of the data that will be put in the validation set
        :param reference: reference in the dataset.json file
        :param licence: licence in the dataset.json file
        :param modalities: dictionary of the modalities available for each patient
        :param rois_dict: Dictionary of the int associated to each ROI's normalized name in the label map (ground truth)
        :param aliases_dict: Dictionary of the aliases for each ROI's normalized name.
        See the ROIDictBuilder class for details on the dictionary's format.
        :param nn_unet_version: the targeted version of nnUNet. Allows the builder to fit the required format for the
        target nnUnet version. In particular, modifies the way the dataset.json.
        """
        if modalities is None:
            modalities = {"0": "CT"}
        self.sorter = DICOMSorter(src_dir=src_dir, dest_dir=inter_dir, logger=logger, force_redo_copy=force_redo_copy)
        self.nn_unet_version = nn_unet_version
        self.task_id = task_id
        self.task_name = task_name
        if self.nn_unet_version == 1:
            self.main_folder = f"Task_{str(self.task_id).zfill(3)}"
        elif self.nn_unet_version == 2:
            self.main_folder = f"Dataset{str(self.task_id).zfill(3)}_{self.task_name}"
        if 1 > self.task_id > 999:
            raise ValueError(f"Task id must be between 1 and 999, found {self.task_id}")
        self.dest_dir = dest_dir
        self.logger = logger
        self.validation_split = validation_split
        if not 0 < self.validation_split < 1:
            raise ValueError(f"Validation split must be between 0 and 1, found {self.validation_split}")
        self.reference = reference
        self.licence = licence
        self.release = release
        self.modalities = modalities
        self.patient_number_to_name_dict = {}
        self.count_ = 0
        self.rois_dict = rois_dict
        self.aliases_dict = aliases_dict
        self.images_test_paths = []
        self.images_train_paths = []
        self.logger = logger

    def _make_dest_dir(self):
        """
        Creates the desired directories' architecture in the destination folder

        This function is private and should not be used outside the MedDecathlonFolderBuilderFromDICOM class
        (unless you know what you are doing)
        :return: None
        """
        maybe_mkdir(self.dest_dir)
        maybe_mkdir(f"{self.dest_dir}{os.sep}{self.main_folder}")
        maybe_mkdir(f"{self.dest_dir}{os.sep}{self.main_folder}{os.sep}imagesTr")
        maybe_mkdir(f"{self.dest_dir}{os.sep}{self.main_folder}{os.sep}imagesTs")
        maybe_mkdir(f"{self.dest_dir}{os.sep}{self.main_folder}{os.sep}labelsTr")
        maybe_mkdir(f"{self.dest_dir}{os.sep}{self.main_folder}{os.sep}labelsTs")

    def _make_train_test(self):
        """
        Splits the data in train / test subsets and handles the transformation of patients' data from DICOM to nii.gz

        This function is private and should not be used outside the MedDecathlonFolderBuilderFromDICOM class
        (unless you kow what you are doing)
        :return:
        """
        nb_patients = len(list(self.sorter.patients.keys()))
        logger.debug(f"Found {nb_patients} patients")
        tt_split_idx = round(nb_patients * self.validation_split)
        self.train_split = list(self.sorter.patients.keys())[:tt_split_idx]
        self.test_split = list(self.sorter.patients.keys())[tt_split_idx:]
        self.DICOM_to_nii(list_patients=self.train_split, src_dir=self.sorter.dest_dir,
                          dest_dir_images=f"{self.dest_dir}{os.sep}{self.main_folder}{os.sep}imagesTr",
                          dest_dir_labels=f"{self.dest_dir}{os.sep}{self.main_folder}{os.sep}labelsTr",
                          is_train=True)

        self.DICOM_to_nii(list_patients=self.test_split, src_dir=self.sorter.dest_dir,
                          dest_dir_images=f"{self.dest_dir}{os.sep}{self.main_folder}{os.sep}imagesTs",
                          dest_dir_labels=f"{self.dest_dir}{os.sep}{self.main_folder}{os.sep}labelsTs",
                          is_train=False)

    def _make_dataset_JSON(self):  # NOQA
        """
        Creates the dataset.json file needed for the task

        This function is private and should not be used outside the MedDecathlonFolderBuilderFromDICOM class
        (unless you kow what you are doing)
        :return: None
        """
        print("\r[+] Writing dataset.json file", end="")
        file_path = f"{self.dest_dir}{os.sep}{self.main_folder}{os.sep}dataset.json"
        f = open(file_path, "w")
        f.write("{\n")
        f.write(f" \"name\": \"{self.task_name}\",\n")
        f.write(f" \"Description\": \"{self.task_name}\",\n")
        if self.nn_unet_version == 1:
            f.write(" \"tensorImageSize\": \"3D\",\n")
        f.write(f" \"reference\": \"{self.reference}\",\n")
        f.write(f" \"licence\": \"{self.licence}\",\n")
        f.write(f" \"release\": \"{self.release}\",\n")
        if self.nn_unet_version == 1:
            f.write(" \"modality\":  {\n")
        elif self.nn_unet_version == 2:
            f.write(" \"channel_names\":  {\n")
        for modality in list(self.modalities.keys())[:-1]:  # NOQA
            f.write(f"    \"{modality}\" : \"{self.modalities[modality]}\",\n")
        f.write(f"    \"{list(self.modalities.keys())[-1]}\" : "
                f"\"{self.modalities[list(self.modalities.keys())[-1]]}\"\n")
        f.write(" },\n")
        f.write(" \"labels\":  {\n")
        for label in list(self.rois_dict.keys())[:-1]:  # NOQA
            if self.nn_unet_version == 1:
                f.write(f"    \"{self.rois_dict[label]}\" : \"{label}\",\n")
            elif self.nn_unet_version == 2:
                f.write(f"    \"{label}\" : \"{self.rois_dict[label]}\",\n")
        if self.nn_unet_version == 1:
            f.write(f"    \"{self.rois_dict[list(self.rois_dict.keys())[-1]]}\" : \"{list(self.rois_dict.keys())[-1]}\"\n")
        elif self.nn_unet_version == 2:
            f.write(f"    \"{list(self.rois_dict.keys())[-1]}\" : \"{self.rois_dict[list(self.rois_dict.keys())[-1]]}\"\n")
        f.write(" },\n")
        f.write(f" \"numTraining\": {len(self.train_split)},\n")
        if self.nn_unet_version == 2:
            f.write(f" \"file_ending\": \".nii.gz\",\n")
        f.write(f" \"numTest\": {len(self.test_split)},\n")
        if self.nn_unet_version == 1:
            f.write(f" \"training\": {json.dumps(self.images_train_paths)},\n")
            f.write(f" \"test\": {json.dumps(self.images_test_paths)}\n")
        f.write(" }")
        f.close()
        print("Done")

    def DICOM_to_nii(self, list_patients: List[str], src_dir: str, dest_dir_images: str, dest_dir_labels: str,  # NOQA
                     is_train: bool):
        """
        This function handles the transformation of DICOM based patients to nii.gz based patients
        (compressed NifTi file format).
        :param list_patients: list of all patients to be processed
        :param src_dir: directory in which the patients' folders can be found.
        :param dest_dir_images: destination directory for the input images
        :param dest_dir_labels: destination directory for the corresponding labels
        :param is_train:bool: is a list of patients used for training. If false, only the input images paths
        will be added to the dataset.json file
        :return:
        """
        for patient in tqdm(list_patients):
            try:
                self.count_ += 1
                if len(self.modalities) > 1:
                    for modality in self.modalities.keys():
                        reader = sitk.ImageSeriesReader()
                        self.make_imgs_one_modality(reader, src_dir, patient, dest_dir_images, dest_dir_labels,
                                                    str(modality), is_train)

                else:
                    reader = sitk.ImageSeriesReader()
                    self.make_imgs_one_modality(reader, src_dir, patient, dest_dir_images, dest_dir_labels,
                                                "0", is_train)

            except:  # NOQA 722
                self.count_ -= 1
                self.logger.error(traceback.format_exc())

    def make_imgs_one_modality(self, reader, src_dir: str, patient: str, dest_dir_images: str, dest_dir_labels: str,
                               modality_nb: str, is_train: bool):
        series_uid = check_dcm_series(f"{src_dir}{os.sep}{patient}{os.sep}"
                                      f"{self.modalities[list(self.modalities.keys())[0]]}",
                                      logger=self.logger)
        dicom_names = reader.GetGDCMSeriesFileNames(f"{src_dir}{os.sep}{patient}{os.sep}"
                                                    f"{self.modalities[list(self.modalities.keys())[0]]}",
                                                    seriesID=series_uid)  # .encode('utf-8'))
        clear_wrong_uids(f"{src_dir}{os.sep}{patient}{os.sep}"
                         f"{self.modalities[list(self.modalities.keys())[0]]}", correct_uid=series_uid)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        nii_name = f"{dest_dir_images}{os.sep}{self.task_name}_{str(self.count_).zfill(3)}.nii.gz"
        nii_path = f"{dest_dir_images}{os.sep}{self.task_name}_{str(self.count_).zfill(3)}" \
                   f"_{modality_nb.zfill(4)}.nii.gz"
        sitk.WriteImage(image, nii_path)
        nii_label_name = f"{dest_dir_labels}{os.sep}{self.task_name}_{str(self.count_).zfill(3)}.nii.gz"
        nii_label_path = f"{dest_dir_labels}{os.sep}{self.task_name}_{str(self.count_).zfill(3)}.nii.gz"

        pdtn = PatientDICOMRTToNII(patient_path=f"{src_dir}{os.sep}{patient}",
                                   rois_dict=self.rois_dict, aliases_dict=self.aliases_dict, ct_nii_path=nii_path,
                                   logger=self.logger, series_uid=series_uid, strict_sop_matching=False)
        pdtn.save_nii()
        try:
            shutil.move(f"{src_dir}{os.sep}{patient}{os.sep}SEG{os.sep}seg.nii.gz", nii_label_path)
            if is_train:
                self.images_train_paths.append(
                    {"image": nii_name, "label": nii_label_name})  # NOQA nii_path is defined anyway
            else:
                self.images_test_paths.append(nii_name)
            self.patient_number_to_name_dict[f"{self.task_name}_{str(self.count_).zfill(3)}"] = patient
        except FileNotFoundError:
            os.remove(nii_path)
            print(traceback.format_exc())
            self.count_ -= 1

        # TODO implement this properly

    def save_patient_correspondance(self):
        """
        Saves the patient and corresponding MSD number
        :return:
        """
        fname = f"{self.main_folder}_correspondance.csv"
        f = open(fname, "w")
        f.write("Number,Patient\n")
        for key in self.patient_number_to_name_dict:
            f.write(f"{key},{self.patient_number_to_name_dict[key]}")
            f.write("\n")
        f.close()

    def call(self):
        """
        Main function for the MedDecathlonFolderBuilderFromDICOM class. Launches the full treatment pipeline.
        :return:
        """
        self._make_dest_dir()
        self.sorter.process_all_dicoms()
        self._make_train_test()
        self._make_dataset_JSON()
        self.save_patient_correspondance()
        print(self.patient_number_to_name_dict)


class ROIDictBuilder:
    """
    This builder allows to create a dictionary of aliases for ROIs.
    This can be useful if the manual segmentations were done without standardized ROIs nas, leading to the same ROI
    being given different names.
    """
    def __init__(self, file_to_parse: str, save_file: str = None):
        self.save_file = save_file
        self.filepath = file_to_parse
        self.rois_dict = {}
        self._build_dict()

    def _build_dict(self) -> None:
        """
        Builds ROIs' dictionary
        """
        with open(self.filepath, "r") as f:
            line = f.readline()
            while not line == "":
                if line[0] == "\\":
                    normalized_name = line.split("\\")[-1][:-1]
                elif line == "\n":
                    pass  # Empty line
                else:
                    key = line.split("\\")[0].replace("\n", "")
                    self.rois_dict[key.lower()] = normalized_name
                line = f.readline()

    def get_dict(self) -> dict[str, str]:
        """
        Getter for the rois' dictionary
        :return: dict[str, str]: A dictionary with aliases as keys, and the corresponding normalized name as value
        """
        return self.rois_dict

    def save_dict(self) -> None:
        """
        Saves dictionary using the pickle module
        """
        pickle.dump(self.rois_dict, open(self.save_file, "wb"))


class MedicalDecathlonToNPY:
    def __init__(self, md_path: str, dest_dir: str, task_id: int, dest_spacing=None):
        self.md_path = md_path
        self.dest_dir = dest_dir
        self.task_id = task_id
        self.dest_spacing = dest_spacing
        if self.dest_spacing is not None:
            assert check_spacing_ok(self.dest_spacing), f"Spacing must be a float or a tuple of three floats " \
                                                        f"{self.dest_spacing}"
            self.resampling_transform_image = tio.Compose([tio.Resample(dest_spacing, image_interpolation="bspline"),
                                                           tio.ZNormalization()])
            self.resampling_transform_mask = tio.Resample(dest_spacing, image_interpolation="nearest")
        self.images_tr_path = f"{md_path}{os.sep}Task{str(task_id).zfill(3)}{os.sep}imagesTr"
        self.images_ts_path = f"{md_path}{os.sep}Task{str(task_id).zfill(3)}{os.sep}imagesTs"
        self.labels_tr_path = f"{md_path}{os.sep}Task{str(task_id).zfill(3)}{os.sep}labelsTr"
        self.labels_ts_path = f"{md_path}{os.sep}Task{str(task_id).zfill(3)}{os.sep}labelsTs"
        self.images_tr_path_dest = f"{dest_dir}{os.sep}Task{str(task_id).zfill(3)}{os.sep}imagesTr"
        self.images_ts_path_dest = f"{dest_dir}{os.sep}Task{str(task_id).zfill(3)}{os.sep}imagesTs"
        self.labels_tr_path_dest = f"{dest_dir}{os.sep}Task{str(task_id).zfill(3)}{os.sep}labelsTr"
        self.labels_ts_path_dest = f"{dest_dir}{os.sep}Task{str(task_id).zfill(3)}{os.sep}labelsTs"

    def make_dirs(self):
        maybe_mkdir(self.dest_dir)
        maybe_mkdir(f"{self.dest_dir}{os.sep}Task{str(self.task_id).zfill(3)}")
        maybe_mkdir(self.images_tr_path_dest)
        maybe_mkdir(self.images_ts_path_dest)
        maybe_mkdir(self.labels_tr_path_dest)
        maybe_mkdir(self.labels_ts_path_dest)

    def move(self):
        # Move train
        train_images_list = sorted(os.listdir(self.images_tr_path))
        for nii_name in train_images_list:
            label_name = f"{nii_name[:-12]}.nii.gz"
            dest_im_name = f"{self.images_tr_path_dest}{os.sep}{nii_name[:-7]}.npy"
            dest_im_a_name = f"{self.images_tr_path_dest}{os.sep}{nii_name[:-7]}_affine.npy"
            dest_label_name = f"{self.labels_tr_path_dest}{os.sep}{label_name[:-7]}.npy"
            dest_label_a_name = f"{self.labels_tr_path_dest}{os.sep}{label_name[:-7]}_affine.npy"
            if os.path.exists(dest_label_name) and os.path.exists(dest_im_name):
                pass
            else:
                pnii2npy = PatientNII2NPY(f"{self.images_tr_path}{os.sep}{nii_name}",
                                          f"{self.labels_tr_path}{os.sep}{label_name}",
                                          True, self.resampling_transform_image)  # TODO handle multiple modalities
                try:
                    pnii2npy.process()
                    image_npy_name = f"{self.images_tr_path}{os.sep}{nii_name[:-7]}.npy"
                    image_affine_npy_name = f"{self.images_tr_path}{os.sep}{nii_name[:-7]}_affine.npy"
                    label_npy_name = f"{self.labels_tr_path}{os.sep}{label_name[:-7]}.npy"
                    label_affine_npy_name = f"{self.labels_tr_path}{os.sep}{label_name[:-7]}_affine.npy"
                    shutil.move(image_npy_name, dest_im_name)
                    shutil.move(image_affine_npy_name, dest_im_a_name)
                    shutil.move(label_npy_name, dest_label_name)
                    shutil.move(label_affine_npy_name, dest_label_a_name)
                except AffinesNotMatchingError:
                    print(f"{nii_name} failed : affines did not match")
                del pnii2npy

        # Move test
        test_images_list = sorted(os.listdir(self.images_ts_path))
        for nii_name in test_images_list:
            label_name = f"{nii_name[:-12]}.nii.gz"
            pnii2npy = PatientNII2NPY(f"{self.images_ts_path}{os.sep}{nii_name}",
                                      f"{self.labels_ts_path}{os.sep}{label_name}",
                                      True, self.resampling_transform_image)  # TODO handle multiple modalities
            try:
                pnii2npy.process()
                image_npy_name = f"{self.images_ts_path}{os.sep}{nii_name[:-7]}.npy"
                image_affine_npy_name = f"{self.images_ts_path}{os.sep}{nii_name[:-7]}_affine.npy"

                dest_im_a_name = f"{self.images_ts_path_dest}{os.sep}{nii_name[:-7]}_affine.npy"
                dest_label_a_name = f"{self.labels_ts_path_dest}{os.sep}{label_name[:-7]}_affine.npy"

                label_npy_name = f"{self.labels_ts_path}{os.sep}{label_name[:-7]}.npy"
                label_affine_npy_name = f"{self.labels_ts_path}{os.sep}{label_name[:-7]}_affine.npy"

                shutil.move(image_npy_name, f"{self.images_ts_path_dest}{os.sep}{nii_name[:-7]}.npy")
                shutil.move(label_npy_name, f"{self.labels_ts_path_dest}{os.sep}{label_name[:-7]}_0000.npy")
                shutil.move(image_affine_npy_name, dest_im_a_name)
                shutil.move(label_affine_npy_name, dest_label_a_name)
                print(image_affine_npy_name)
                print(label_affine_npy_name)
            except AffinesNotMatchingError:
                print(f"{nii_name} failed : affines did not match")
            del pnii2npy

    def process(self):
        self.make_dirs()
        self.move()


class NPY2MSD:
    def __init__(self, src_path, dst_path, task_id: int, rois_dict: dict, modalities: dict, task_name: str,
                 reference: str, release: str, licence: str,
                 spacing: Optional[Union[float, Tuple[float, float, float]]] = 1, nnunet_version: int = 2,
                 test_split_ratio : float = .2):
        self.src_path = src_path
        self.dst_path = dst_path
        self.task_id = task_id
        self.rois_dict = rois_dict
        self.modalities = modalities
        self.task_name = task_name
        self.release = release
        self.reference = reference
        self.licence = licence
        self.images_train_paths = []
        self.images_test_paths = []
        if isinstance(spacing, float):
            self.spacing = (spacing, spacing, spacing)
        elif isinstance(spacing, int):
            self.spacing = (float(spacing), float(spacing), float(spacing))
        else:
            self.spacing = spacing
        self.nn_unet_version = nnunet_version
        self.test_split_ratio = test_split_ratio
        self.nb_patients = len(self._get_list_patients())
        self.test_split_idx = int(round(self.nb_patients*self.test_split_ratio))
        self.test_len =self.nb_patients - self.test_split_idx

    def _get_list_patients(self):
        return sorted(os.listdir(self.src_path))

    def _get_paths_dict(self):
        dict_out = {}
        for i, patient_name in enumerate(self._get_list_patients()):
            for key in self.modalities.keys():
                mod_path_src = os.path.join(self.src_path , patient_name, f"{patient_name}_{self.modalities[key]}")
                mod_path_dst = f"{self.task_name}_{i}_{self.modalities[key].zfill(4)}.nii.gz"
                dict_out[mod_path_src] = mod_path_dst
        return dict_out

    def _get_affine(self):
        aff = np.eye(4)
        aff[0][0] = self.spacing[0]
        aff[1][1] = self.spacing[1]
        aff[2][2] = self.spacing[2]
        return aff

    def _make_dst_dirs(self):
        """
        Creates the desired directories' architecture in the destination folder

        This function is private and should not be used outside the MedDecathlonFolderBuilderFromDICOM class
        (unless you know what you are doing)
        :return: None
        """
        maybe_mkdir(self.dst_path)
        maybe_mkdir(f"{self.dst_path}{os.sep}Task{str(self.task_id).zfill(3)}")
        maybe_mkdir(f"{self.dst_path}{os.sep}Task{str(self.task_id).zfill(3)}{os.sep}imagesTr")
        maybe_mkdir(f"{self.dst_path}{os.sep}Task{str(self.task_id).zfill(3)}{os.sep}imagesTs")
        maybe_mkdir(f"{self.dst_path}{os.sep}Task{str(self.task_id).zfill(3)}{os.sep}labelsTr")
        maybe_mkdir(f"{self.dst_path}{os.sep}Task{str(self.task_id).zfill(3)}{os.sep}labelsTs")


    def _make_dataset_JSON(self):  # NOQA
        """
        Creates the dataset.json file needed for the task

        This function is private and should not be used outside the MedDecathlonFolderBuilderFromDICOM class
        (unless you kow what you are doing)
        :return: None
        """
        print("\r[+] Writing dataset.json file", end="")
        file_path = f"{self.dst_path}{os.sep}Task{str(self.task_id).zfill(3)}{os.sep}dataset.json"
        f = open(file_path, "w")
        f.write("{\n")
        f.write(f" \"name\": \"{self.task_name}\",\n")
        f.write(f" \"Description\": \"{self.task_name}\",\n")
        f.write(" \"tensorImageSize\": \"3D\",\n")
        f.write(f" \"reference\": \"{self.reference}\",\n")
        f.write(f" \"licence\": \"{self.licence}\",\n")
        f.write(f" \"release\": \"{self.release}\",\n")
        if self.nn_unet_version == 1:
            f.write(" \"modality\":  {\n")
        elif self.nn_unet_version == 2:
            f.write(" \"channel_names\":  {\n")
        for modality in list(self.modalities.keys())[:-1]:  # NOQA
            f.write(f"    \"{modality}\" : \"{self.modalities[modality]}\",\n")
        f.write(f"    \"{list(self.modalities.keys())[-1]}\" : "
                f"\"{self.modalities[list(self.modalities.keys())[-1]]}\"\n")
        f.write(" },\n")
        f.write(" \"labels\":  {\n")
        for label in list(self.rois_dict.keys())[:-1]:  # NOQA
            if self.nn_unet_version == 1:
                f.write(f"    \"{self.rois_dict[label]}\" : \"{label}\",\n")
            elif self.nn_unet_version == 2:
                f.write(f"    \"{label}\" : \"{self.rois_dict[label]}\",\n")
        if self.nn_unet_version == 1:
            f.write(f"    \"{self.rois_dict[list(self.rois_dict.keys())[-1]]}\" : \"{list(self.rois_dict.keys())[-1]}\"\n")
        elif self.nn_unet_version == 2:
            f.write(f"    \"{list(self.rois_dict.keys())[-1]}\" : \"{self.rois_dict[list(self.rois_dict.keys())[-1]]}\"\n")
        f.write(" },\n")
        f.write(f" \"numTraining\": {self.test_split_idx},\n")
        if self.nn_unet_version == 2:
            f.write(f" \"file_ending\": \".nii.gz\"")
        f.write(f" \"numTest\": {self.test_len},\n")
        f.write(f" \"training\": {json.dumps(self.images_train_paths)},\n")
        f.write(f" \"test\": {json.dumps(self.images_test_paths)}\n")
        f.write(" }")
        f.close()
        print("Done")

    def process(self):
        paths_dict = self._get_paths_dict()
        for i, key in enumerate(paths_dict.keys()):
            src = key
            if i < self.test_split_idx:
                dst = os.path.join(self.dst_path, "imagesTr", paths_dict[key])
                self.images_train_paths.append(dst)
            else:
                dst = os.path.join(self.dst_path, "imagesTs", paths_dict[key])
                self.images_test_paths.append(dst)


if __name__ == "__main__":
    LOG_FORMAT = "%(levelname)s %(asctime)s %(message)s"
    logging.basicConfig(filename="mediproc_only_flap.log", level=logging.DEBUG, format=LOG_FORMAT)
    logger = logging.getLogger()
    logger.info(sys.argv[0])

    warnings.filterwarnings("ignore")
    # In ths dictionary, define which value should be assigned to each label (normalized names only)
    # Since a voxel can only be attributed one value, if two RT structs contours overlap, the one with smaller label
    # will be kept. You should attribute the values to labels in order of importance, excepted for background which
    # should be zero
    """labels_dict = {"Background": 0, "FLAP": 1, "Boneflap": 2, "Brainstem": 3, "Mandibule": 4, "Glandes parotides": 5,
                   "Moelle épinière": 6, "Canal medullaire": 7, "Chiasme": 8, "Cochlée": 9, "Crystallin": 10,
                   "Oeil": 11, "Nerf optique": 12}"""
    # labels_dict = {"background": 0, "FLAP": 1}
    labels_dict = {"background": 0, "FLAP": 1, "Tronc cérébral": 2, "Encéphale": 3, "Moelle épinière": 4,
                   "Canal médullaire": 5, "Trachée": 6, "Thyroïde": 7, "Oesophage": 8, "Oeil D": 9, "Oeil G": 10,
                   "Mandibule": 11, "Lèvres": 12, "Poumon D": 13, "Poumon G": 14, "Larynx": 15,
                   "Constr_Pharynx": 16}
    aliases_dict_builder = ROIDictBuilder("../FLAP/Interesting_rois.txt", save_file="dict_rois.pckl")
    aliases_dict_builder.save_dict()
    aliases_dict = pickle.load(open("dict_rois.pckl", "rb"))
    # aliases_dict = aliases_dict_builder.get_dict()
    if sys.argv[1] == "msd":
        msd_builder = MedDecathlonFolderBuilderFromDICOM(task_id=511, task_name="FLAP_MOAR",
                                                          dest_dir="/mnt/disk_2/Zach/FLAP_MOAR_DICOM_MSD",
                                                         inter_dir="/mnt/disk_2/Zach/FLAP_MOAR_DICOM_SORTED",
                                                         src_dir="/mnt/disk_2/Zach/FLAP_DICOM",
                                                         rois_dict=labels_dict,
                                                         aliases_dict=aliases_dict, logger=logger,
                                                         modalities={0: "CT"},
                                                         reference="CHB Rouen", licence="MIT", release="1.0 03/00/2023",
                                                         validation_split=.8, force_redo_copy=False)
        msd_builder.call()

    elif sys.argv[1] == "npy":
        # mds_path = "/mnt/disk_2/Zach/nnunet_db/nnUNet_raw_data"
        mds_path = "/mnt/disk_2/Zach/XFLAP_MSD"
        npy_path = "/mnt/disk_2/Zach/XFLAP_NPY_RESAMPLED"
        m2npy = MedicalDecathlonToNPY(mds_path, npy_path, 603, (1.1696, 1.1696, 2.))
        m2npy.process()

    elif len(sys.argv) == 3:
        # Sorting only
        path_in = sys.argv[1]
        path_out = sys.argv[2]
        ds = DICOMSorter(src_dir=path_in, dest_dir=path_out, logger=logger)
        ds.process_all_dicoms()

        drtn = DirectoryDICOMToNRRD(path_out, rois_dict=labels_dict, aliases_dict=aliases_dict,
                                    logger=logger, error_if_mismatched_RT_SOP_UID=False)
        drtn.process(save_compressed_nii=True)
    elif sys.argv[1] == "nii_one":
        path = sys.argv[1]

# python3 main.py

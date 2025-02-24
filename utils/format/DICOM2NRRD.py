import os
import traceback
import warnings
from typing import List, Union

import nrrd
import numpy as np
import pydicom
from rt_utils import RTStruct, RTStructBuilder
import SimpleITK as sitk


class PatientDICOMToNRRD:
    """
    This class handles a patient for which DICOM files ( CT storage and RT structure storage) are available.
    Using the [rt-utils](https://github.com/qurit/rt-utils) module, this class extracts a segmentation mask in
    the form of a numpy array frm the RT structure set storage files. It then saves this mask to Nearly Raw Raster Data
    (NRRD) file, while conserving the reference patient space.
    """
    def __init__(self, patient_path, rois_dict: dict[str, int], aliases_dict: dict[str, str], logger,
                 error_if_mismatched_RT_SOP_UID: bool = True):  # NOQA
        self.path = patient_path
        self.slice_thickness_tag = pydicom.tag.Tag('0018', '0050')
        self.pixel_spacing_tag = pydicom.tag.Tag('0028', '0030')
        self.image_position_patient_tag = pydicom.tag.Tag('0020', '0032')
        self.ct_dir = f"{self.path}{os.sep}CT"
        self.rt_struct_dir = f"{self.path}{os.sep}RTStruct"
        self.rois_dict = rois_dict
        self.aliases_dict = aliases_dict
        self.logger = logger
        self.dims = None
        self.error_if_mismatched_RT_SOP_UID = error_if_mismatched_RT_SOP_UID

    def get_origin(self) -> List:
        """
        Gets the  origin coordinates from the CT scan
        :return List
        """
        slice_orig = 1000000
        dicom_final = None
        for dicom_file in os.listdir(self.ct_dir):
            dicom = pydicom.dcmread(f"{self.ct_dir}{os.sep}{dicom_file}")
            temp = dicom[self.image_position_patient_tag].value[-1]
            if temp < slice_orig:
                slice_orig = temp
                dicom_final = dicom
        image_position_patient = dicom_final[self.image_position_patient_tag].value
        origin = [float(image_position_patient[0]), float(image_position_patient[1]), float(slice_orig)]
        return origin

    def get_space(self) -> str:  # NOQA
        """
        Returns the definition of the space in which our datsa exists.
        It seems the space used by the DICOM stanard corresponds to the "LPS" space definition in the NRRD standard.
        :return: str
        """
        # TODO: remove the "it seems" bit once we're sure
        return "LPS"

    def get_spacing(self) -> List:
        """
        Gets the spacing (i.e.: voxel dimensions from the CT scan.
        The values are already formatted for the "space directions" field in a NRRD header:
        [[spacing_x, 0,0], [0, spacing_y, 0], [0, 0, slice_thickness]]
        :return: List
        """
        dicom_ = pydicom.dcmread(f"{self.ct_dir}{os.sep}{os.listdir(self.ct_dir)[1]}")
        thickness = dicom_[self.slice_thickness_tag]
        spacings = dicom_[self.pixel_spacing_tag]
        spacings_list = [[float(spacings[0]), 0, 0], [0, float(spacings[1]), 0], [0, 0, float(thickness.value)]]
        return spacings_list

    def _load_rt_file(self, rt_file_name) -> Union[RTStruct, None]:
        """
        Loads a RT file, checking for its validity and compatibility with the CT scan along the way.
        :param rt_file_name: Name of the RTStruct file to load
        :return: RTStruct if the RTStruct could be loaded by rt-utils
        """
        try:
            rtstruct = RTStructBuilder.create_from(
                dicom_series_path=self.ct_dir,
                rt_struct_path=f"{self.rt_struct_dir}{os.sep}{rt_file_name}",
                warn_only=not self.error_if_mismatched_RT_SOP_UID
            )
            return rtstruct
        except Exception as e:
            self.logger.warning(f"{rt_file_name} triggered exception {e}. This might cause an error.")
            return None

    def get_mask(self) -> np.array:
        """
        Gets the segmentation mask as a numpy array. The mask is extracted from the DICOM RT files using rt-utils.
        If not at least one ROI with a name present in the aliases' dictionary is found, an Exception is raised
        :return: np.array: the segmentation mask
        :raises: Exception
        """
        seg_mask = None
        rts_list = os.listdir(self.rt_struct_dir)
        normalized_names = []
        for rt_file in rts_list:
            rtstruct = self._load_rt_file(rt_file)
            if rtstruct is None:
                pass
            else:
                for name_ in rtstruct.get_roi_names():
                    if name_.lower() in self.aliases_dict.keys():
                        normalized_name = self.aliases_dict[name_.lower()]
                        normalized_names.append(normalized_name)
                        if normalized_name in self.rois_dict.keys():
                            label_value = self.rois_dict[normalized_name]
                            mask_3d = rtstruct.get_roi_mask_by_name(name_) * 1
                            mask_3d = np.transpose(mask_3d, (1, 0, 2))
                            sizes = np.shape(mask_3d)
                            dims = [len(self.rois_dict.keys())] + list(sizes)
                            if seg_mask is None:
                                seg_mask = np.zeros(dims)
                            seg_mask[label_value] = np.logical_or(mask_3d, seg_mask[label_value])
        if seg_mask is None:
            msg = f"No ROI from the dictionary was found for patient {self.path}. Maybe an alias is missing" \
                  f" in your dictionary, or no valid RT file was found." \
                  f"List of normalized names found : {normalized_names}"
            warnings.warn(msg)
        try:
            self.dims = dims[1:]  # NOQA
        except NameError:
            return None  # dims not defined, no ROI was found
        seg_mask = np.argmax(seg_mask, axis=0, keepdims=False)
        return seg_mask

    def get_header(self) -> dict:
        """
        Builds and returns a NRRD header
        :return: dict: NRRD Header
        """
        return {"space": self.get_space(), 'space origin': np.array(self.get_origin()), 'space units': ['mm', 'mm', 'mm'],
                'sizes': self.dims, 'space directions': self.get_spacing(), 'encoding': 'ASCII', 'type': 'uchar',
                'dimension': 3}

    def save_nrrd(  self, save_compressed_nii=False) -> None:
        """
        Saves the NRRD file to disk.
        :param save_compressed_nii: bool : if True, saves using the compressed Nifti (nii.gz) file format as well
        """
        try:
            seg_mask = self.get_mask()
            if seg_mask is None:
                self.logger.warning("Not saving a nrrd file. Se warnings above for a possible explanation")
                return
            header = self.get_header()
            try:
                os.mkdir(f"{self.path}{os.sep}SEG")
            except FileExistsError:
                pass
            nrrd.write(f"{self.path}{os.sep}SEG{os.sep}seg.nrrd", seg_mask, header)
            if save_compressed_nii:
                nrrd_path = f"{self.path}{os.sep}SEG{os.sep}seg.nrrd"
                nii_path = f"{self.path}{os.sep}SEG{os.sep}seg.nii.gz"
                img_nrrd = sitk.ReadImage(nrrd_path)
                sitk.WriteImage(img_nrrd, nii_path)
                os.remove(nrrd_path)
        except:   # NOQA: 772
            self.logger.error(f"{self.path} failed. See error below.")
            self.logger.error(traceback.format_exc())

    def process(self, save_compressed_nii=False):
        """
        Alias for the save_nrrd function. This function handles the full transformation of a patient from DICOM to NRRD,
        with the nii.gz compression if specified.
        :param save_compressed_nii:
        :return:
        """
        self.save_nrrd(save_compressed_nii)


class DirectoryDICOMToNRRD:
    """This class handles the processing of a directory full of DICOM patients
    (patients' directories created by the DicomSorter class)"""
    def __init__(self, dir_path, rois_dict, aliases_dict, logger, error_if_mismatched_RT_SOP_UID: bool = True):  # NOQA
        """
        :param dir_path: Path to the source directory
        :param rois_dict: Dictionary of the int associated to each ROI's normalized name in the label map (ground truth)
        :param aliases_dict: Dictionary of the aliases for each ROI's normalized name.
        :param logger: Logger to log errors and warnings
        """
        self.path = dir_path
        self.list_patients = sorted(os.listdir(self.path))
        self.rois_dict = rois_dict
        self.aliases_dict = aliases_dict
        self.logger = logger
        self.error_if_mismatched_RT_SOP_UID = error_if_mismatched_RT_SOP_UID

    def process(self, save_compressed_nii=False):
        """

        :param save_compressed_nii: bool : Whether images should be compressed and saved to nii.gz format as well.
        :return:
        """
        for patient in self.list_patients:
            prtn = PatientDICOMToNRRD(f"{self.path}{os.sep}{patient}", rois_dict=self.rois_dict,
                                      aliases_dict=self.aliases_dict, logger=self.logger,
                                      error_if_mismatched_RT_SOP_UID=self.error_if_mismatched_RT_SOP_UID)
            try:
                prtn.process(save_compressed_nii)
            except Exception as e:
                self.logger.error(f"{patient} failed : {traceback.format_exc()} at {e.__traceback__.tb_lineno}")

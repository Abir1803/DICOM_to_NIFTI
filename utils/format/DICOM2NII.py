import logging
import traceback
import warnings
from typing import Union

import pydicom
import os
import numpy as np
from rt_utils import RTStruct, RTStructBuilder
import nibabel as nib


class PatientDICOMRTToNII:
    """
    This class handles a patient for which RT structure storage DICOM files are available.
    Using the [rt-utils](https://github.com/qurit/rt-utils) module, this class extracts a segmentation mask in
    the form of a numpy array frm the RT structure set storage files. It then saves this mask to compressed Nifti
    (nii.gz) file, while conserving the reference patient space.
    """
    def __init__(self, patient_path, rois_dict: dict[str, int], aliases_dict: dict[str, str], ct_nii_path: str,
                 ct_dir_dicom: str = None, logger=None, series_uid=None, strict_sop_matching=True): # NOQA
        self.path = patient_path
        self.strict_sop_matching = strict_sop_matching
        self.slice_thickness_tag = pydicom.tag.Tag('0018', '0050')
        self.pixel_spacing_tag = pydicom.tag.Tag('0028', '0030')
        self.image_position_patient_tag = pydicom.tag.Tag('0020', '0032')
        self.image_orientation_patient_tag = pydicom.tag.Tag('0020', '0037')
        if ct_dir_dicom is None:
            self.ct_dir = f"{self.path}{os.sep}CT"
        else:
            self.ct_dir = ct_dir_dicom
        self.rt_struct_dir = f"{self.path}{os.sep}RTStruct"
        self.rois_dict = rois_dict
        self.aliases_dict = aliases_dict
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        self.series_uid = series_uid
        self.dims = None
        self.affine = None
        self.dr = None
        self.dc = None
        self.slice_thickness = None
        self.ct_nii = nib.load(ct_nii_path)

    def get_affine(self):
        return self.ct_nii.affine

    def get_affine_old(self):
        slice_orig = 1000000
        slice_end = -1000000
        dicom_first = None
        dicom_last = None
        list_dicom_slices = sorted(os.listdir(self.ct_dir))
        uid_ok = True
        for dicom_file in list_dicom_slices:
            dicom = pydicom.dcmread(f"{self.ct_dir}{os.sep}{dicom_file}")
            if self.series_uid is not None:
                local_seriesUID = dicom[pydicom.tag.Tag("0020", "000e")].value
                if self.series_uid == local_seriesUID:
                    uid_ok = True
                else:
                    uid_ok = False
            if uid_ok:
                temp = dicom[self.image_position_patient_tag].value[-1]
                if temp < slice_orig:
                    slice_orig = temp
                    dicom_first = dicom
                if temp > slice_end:
                    slice_orig = temp
                    dicom_last = dicom
        t1_1, t1_2, t1_3 = dicom_first[self.image_position_patient_tag].value
        tn_1, tn_2, tn_3 = dicom_last[self.image_position_patient_tag].value
        f1_2, f2_2, f3_2, f1_1, f2_1, f3_1 = dicom_last[self.image_orientation_patient_tag].value
        nb_slices = len(list_dicom_slices)
        self.slice_thickness = dicom_first[self.slice_thickness_tag].value
        self.dr, self.dc = dicom_first[self.pixel_spacing_tag].value
        affine = np.array([[f1_1 * self.dr, f1_2 * self.dc, (tn_1 - t1_1) / (nb_slices - 1), t1_1],
                           [f2_1 * self.dr, f2_2 * self.dc, (tn_2 - t1_2) / (nb_slices - 1), t1_2],
                           [f3_1 * self.dr, f3_2 * self.dc, (tn_3 - t1_3) / (nb_slices - 1), t1_3],
                           [0, 0, 0, 1]])
        return affine

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
                warn_only=not self.strict_sop_matching)
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
        print(rts_list)
        normalized_names = []
        roi_names = []
        for rt_file in rts_list:
            rtstruct = self._load_rt_file(rt_file)
            if rtstruct is None:
                pass
            else:
                for name_ in rtstruct.get_roi_names():
                    if name_.lower() in self.aliases_dict.keys():
                        normalized_name = self.aliases_dict[name_.lower()]
                        normalized_names.append(normalized_name)
                        roi_names.append(name_)
                        if normalized_name in self.rois_dict.keys():
                            print(name_.lower(), end=",")
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

    def get_header(self):
        return self.ct_nii.header

    def get_header_old(self, mask, affine):
        empty_header = nib.Nifti1Header()
        empty_header["srow_x"] = affine[0]
        empty_header["srow_y"] = affine[1]
        empty_header["srow_z"] = affine[2]
        empty_header["dim"] = [1]*8
        i = 0
        for dim in np.shape(mask):
            empty_header["dim"][i] = dim
            i += 1
        empty_header["qoffset_x"] = affine[0, 3]
        empty_header["qoffset_y"] = affine[1, 3]
        empty_header["qoffset_z"] = affine[2, 3]
        pixdim = [1., self.dr, self.dc, self.slice_thickness, 0., 0., 0., 0.]
        pixdim = np.array(pixdim, dtype=float)
        print(pixdim)
        empty_header["pixdim"] = pixdim
        return empty_header

    def save_nii(self) -> None:
        """
        Saves the Nifti image to a nii.gz file on disk.
        """
        try:
            seg_mask = self.get_mask()
            if seg_mask is None:
                self.logger.warning("Not saving a nii.gz file. Se warnings above for a possible explanation")
                return
            affine = self.get_affine()
            header = self.get_header()
            nii_dir = self.path
            try:
                os.mkdir(f"{nii_dir}{os.sep}SEG")
            except FileExistsError:
                pass
            img = nib.Nifti1Image(seg_mask, affine, header=header)
            nii_path = f"{nii_dir}{os.sep}SEG{os.sep}seg.nii.gz"
            nib.save(img, nii_path)
        except:   # NOQA: 772
            self.logger.error(f"{self.path} failed. See error below.")
            self.logger.error(traceback.format_exc())


class DirectoryDICOMToNII:
    """This class handles the processing of a directory full of DICOM patients
    (patients' directories created by the DicomSorter class)"""
    def __init__(self, dir_path, rois_dict, aliases_dict, logger):  # NOQA
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

    def process(self):
        """
        :return:
        """
        for patient in self.list_patients:
            prtn = PatientDICOMRTToNII(f"{self.path}{os.sep}{patient}", rois_dict=self.rois_dict,
                                       aliases_dict=self.aliases_dict, logger=self.logger)
            try:
                prtn.save_nii()
            except Exception as e:
                self.logger.error(f"{patient} failed : {traceback.format_exc()} at {e.__traceback__.tb_lineno}")

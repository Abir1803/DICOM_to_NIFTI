import os
import pydicom
from pydicom import dcmread


def check_dcm_series(fpath: str, logger=None):
    has_logger = logger is not None
    if not os.path.exists(fpath):
        if has_logger:
            logger.error(f"No such file or directory {fpath}")
        else:
            print(f"No such file or directory {fpath}")
    else:
        if not os.path.isdir(fpath):
            if has_logger:
                logger.error(f"{fpath} is not a directory")
            else:
                print(f"{fpath} is not a directory")
    seriesUIDs_nb_slices_dict = {}
    for i in os.listdir(fpath):
        dicom_file = dcmread(fpath+os.sep+i)
        seriesUID = dicom_file[pydicom.tag.Tag("0020", "000e")].value
        if seriesUID in seriesUIDs_nb_slices_dict.keys():
            seriesUIDs_nb_slices_dict[seriesUID]+=1
        else:
            seriesUIDs_nb_slices_dict[seriesUID]= 1
    nb_series = len(seriesUIDs_nb_slices_dict.keys())
    if nb_series > 1:
        max_slices = 0
        for uid in seriesUIDs_nb_slices_dict.keys():
            if seriesUIDs_nb_slices_dict[uid] > max_slices:
                uid_used = uid
                max_slices = seriesUIDs_nb_slices_dict[uid]
        logger.warning(f"Found {nb_series} series in directory {fpath}. Using Series with uid {uid_used} and deleting other series.")
    else:
        uid_used = list(seriesUIDs_nb_slices_dict.keys())[0]
    return uid_used


def clear_wrong_uids(folder_path, correct_uid):
    for i in os.listdir(folder_path):
        dicom_file = dcmread(folder_path+os.sep+i)
        series_uid = dicom_file[pydicom.tag.Tag("0020", "000e")].value
        if series_uid != correct_uid:
            os.remove(folder_path+os.sep+i)


def check_spacing_ok(spacing):
    is_ok = True
    if isinstance(spacing, tuple):
        for i in spacing:
            if isinstance(i, float):
                pass
            else:
                is_ok = False
    elif isinstance(spacing, float):
        pass
    else:
        is_ok = False
    return is_ok
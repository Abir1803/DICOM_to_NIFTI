import os
import shutil
import traceback
from os.path import join
import glob

import pydicom
from pydicom.tag import Tag
from .io import maybe_copy
from .io import maybe_mkdir


class DICOMSorter:
    """
    Sorts a directory full of DICOM files by:
     - patient Name
     - file type (CT Scan, RT Struct or RT Plan))
    # TODO: handle RT Plan better (need to learn more about how it works)
    """
    def __init__(self, src_dir, dest_dir=None, logger=None, force_redo_copy=True): # NOQA
        self.src_dir = src_dir
        self.force_redo_copy = force_redo_copy
        if dest_dir is None:
            self.dest_dir = src_dir
        else:
            self.dest_dir = dest_dir
        self.patient_names = set()
        self.patients = {}
        self.PatientName_tag = Tag('0010', '0010')  # NOQA
        self.fileType_tag = Tag('0008', '0060')  # NOQA
        self.logger = logger
        self.modalities = ["CT", "PT", "NM", "MR", "RTStruct", "RTDose", "RTPlan"]

    def get_list_dicoms(self):
        """
        Gets and returns the list of all the DICOMs in src_dir
        and subdirectories (recursive search), sorted alphabeticallyy
        :return: List[str]: List of all the DICOM files names,
        sorted alphabetically
        """
        return sorted([fname for fname in glob.glob(f"{self.src_dir}/**/*",
                                      recursive=True)])

    def process_dicom(self, path):
        """
        Processes one DICOM file. Opens it to read its metadata,
        then saves it in the 'patients' dictionary
        :param path: Path of the DICOM file
        """
        try:
            dicom_ = pydicom.read_file(path, force=True)
            patient_name = dicom_[self.PatientName_tag].value
            #file_type = dicom_.file_meta[self.fileType_tag].value
            #print(file_type)
            if patient_name not in self.patients.keys():
                print(f"[+] Discovered patient {patient_name}")
                self.patients[patient_name] = {}
            if dicom_.Modality in self.modalities:
                if dicom_.Modality not in self.patients[patient_name].keys():
                    self.patients[patient_name][dicom_.Modality] = []
                self.patients[patient_name][modality].append(path.split(os.sep)[-1])  # NOQA
            else:
                self.logger.warning("Found unknown file"
                                    f" type. UI is {dicom_.Modality}")
        except:  # NOQA: 772
            self.logger.error("Error while discovering and "
                              f"integrating file {path}")
            self.logger.error(traceback.format_exc())

    def _discover_all(self) -> None:
        """
        Passes all DICOM file to the process_dicom function.
        This function should not be used outside the DICOMSorter class
        (or one of its children), because it only discovers and registers
        the DICOM files, it does not move them.
        """
        for filename in self.get_list_dicoms():
            self.process_dicom(filename)

    def _copy_all(self) -> None:
        """
        Moves all files to their new location.
        Creates the needed folders:
        (Patient folder, CT / RTStruct / RTDose.. folders)
        """
        copy_fn = shutil.copy if self.force_redo_copy else maybe_copy

        maybe_mkdir(f"{self.dest_dir}")
        for patient in self.patients.keys():
            print(f"[+] Moving files for patient {patient}")
            maybe_mkdir(f"{self.dest_dir}{os.sep}{patient}")
            for modality in self.modalities:
                maybe_mkdir(join(self.dest_dir, patient, modality))

            for modality in self.modalities:
                if modality in self.patients[patient].keys():
                    for file_ct in self.patients[patient][modality]:
                        copy_fn(join(self.src_dir, file_ct),
                                join(self.dest_dir, patient, modality, file_ct))

    def process_all_dicoms(self) -> None:
        """
        Calls the _discover_all and _move_all functions.
        This function should be called. It executes the DICOM sorting
        """
        self.logger.info("______________________Starting sorter______________________")  # NOQA
        self._discover_all()
        print("\n"
              f"[+] Discovered {len(self.patients)} patients"
              )
        self._copy_all()
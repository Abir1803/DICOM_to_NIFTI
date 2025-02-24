import copy
import json
import os
import pickle
import traceback
import warnings
from typing import Union

import SimpleITK as sitk
import nibabel as nib
import inspect

import numpy as np
import torchio
from numpy import floor, ceil

from check.general import each_element_in_list
import radiomics

from check.general import Checker
from data.spatial import space_point_in_mask, euclidean3d, get_spacing, get_origin, get_bb_from_mask





class MSDDataset:
    """
    This class lazily loads an MSD (nnUNet) style dataset, and allows computation of imaging features on it
    Currently implemented:
     - Computation of radiomic features
    """
    def __init__(self, root_path: str, output_folder: str, msd_version: int = 2):
        self .root_dir = root_path
        self.dataset_json_path = os.path.join(root_path, "dataset.json")
        self.msd_version = msd_version
        self.output_folder = output_folder
        self._parse_json()
        self._lazy_load_dataset()
        self.patients_max_coords = {}
        self.patients_min_coords = {}

    def _parse_json(self):
        dataset_json: dict = json.load(open(self.dataset_json_path, "rb"))
        self.name = dataset_json.get("name")
        self.description = dataset_json.get("description")
        self.reference = dataset_json.get("reference")
        self.licence = dataset_json.get("licence")
        self.release = dataset_json.get("release")
        if self.msd_version == 1:
            self.channels = dataset_json.get("modalities")
        elif self.msd_version == 2:
            self.channels = dataset_json.get("channel_names")
        self.nb_channels = len(self.channels.keys())
        self.labels = dataset_json.get("labels")
        if self.msd_version == 1:
            new_dict_labels = {}
            for key in self.labels.keys():
                new_dict_labels[self.labels[key]] = key
            self.labels = new_dict_labels
        self.nb_training = int(dataset_json.get("numTraining"))
        self.nb_testing = dataset_json.get("numTest")
        if self.nb_testing is not None:
            self.nb_testing = int(self.nb_testing)
            self.has_test = bool(self.nb_testing)
        else:
            self.has_test = False
            self.nb_testing = 0

    def _lazy_load_dataset(self):
        if self.msd_version == 1:
            id_ = self.root_dir.split(os.sep)[-1].split("_")[1]
        else :
            id_ = self.root_dir.split(os.sep)[-1].split("_")[0][-3:]
        try:
            self.id = int(id_)
        except:
            raise ValueError(f"Invalid id: {id}")
        self.imagesTr_path = os.path.join(self.root_dir, "imagesTr")
        self.labelsTr_path = os.path.join(self.root_dir, "labelsTr")
        self.imagesTs_path = os.path.join(self.root_dir, "imagesTs")
        self.labelsTs_path = os.path.join(self.root_dir, "labelsTs")
        list_files_training = sorted(os.listdir(self.imagesTr_path))
        list_files_testing = sorted(os.listdir(self.imagesTs_path))

        # Iterating over the training files
        idx = 0
        self.patients_tr = {}
        while idx < len(list_files_training):
            file_name = list_files_training[idx]
            patient_id = file_name.split("_")[1]
            chan = file_name.split(".")[0].split("_")[2] # No need to remove heading path, there is none
            if f"{self.name}_{patient_id}" not in self.patients_tr. keys():
                self.patients_tr[f"{self.name}_{patient_id}"] = {}

            # Maybe remove double cast
            self.patients_tr[f"{self.name}_{patient_id}"][self.channels[str(int(chan))]] = \
                os.path.join(self.root_dir, "imagesTr", file_name)
            idx += 1
        for key in self.patients_tr.keys():
            file_label = self.patients_tr[key][list(self.patients_tr[key].keys())[0]][:-12]\
                .replace(f"{os.sep}imagesTr{os.sep}", f"{os.sep}labelsTr{os.sep}")
            self.patients_tr[key]["label"] = f"{file_label}.nii.gz"

        # Iterating over the testing files
        if self.has_test:
            idx = 0
            self.patients_ts = {}
            while idx < len(list_files_testing):
                file_name = list_files_testing[idx]
                patient_id = file_name.split("_")[1]
                chan = file_name.split(".")[0].split("_")[2] # No need to remove heading path, there is none
                if f"{self.name}_{patient_id}" not in self.patients_ts. keys():
                    self.patients_ts[f"{self.name}_{patient_id}"] = {}
                self.patients_ts[f"{self.name}_{patient_id}"][self.channels[str(int(chan))]]= file_name
                idx += 1

    def compute_radiomics(self,
                          selected_channels: list, radiomics_list: Union[list, tuple],
                          selected_classes_indexes: Union[list[int], tuple[int]], resample_target=None):  # NOQA
        available_raidomics = ("FirstOrder", "Shape", "GLCM", "GLRLM", "GLSZM", "NGTDM", "GLDM")
        is_contained, list_unexpected = each_element_in_list(radiomics_list, available_raidomics,
                                                             return_unexpected_elements=True)

        if not is_contained:
            raise ValueError(f"Found unexpected radiomic name(s): {list_unexpected}. "
                             f"Should be in: {available_raidomics}")
        is_contained, list_unexpected = each_element_in_list(selected_channels, self.channels.values(),
                                                             return_unexpected_elements=True)

        if not is_contained:
            raise ValueError(f"Found unexpected channel name(s): {list_unexpected}. "
                             f"Should be in: {available_raidomics}")
        if resample_target is not None:
            torchio.Resample(resample_target)

        for patient in self.patients_tr.keys():
            for key in selected_channels:
                file = self.patients_tr[patient][key]
                for radiomic in radiomics_list:
                    self.f = open(os.path.join(self.output_folder, f"{patient}_{key}.csv"), "w")
                    print(f"Starting radiomics extraction for {patient}")
                    self._compute_radiomic(file, self.patients_tr[patient]["label"], radiomic, selected_classes_indexes)
                    print("Done radiomics extraction")
                    self.f.close()
                    del self.f

    def _compute_radiomic(self, file, mask, radiomic, indexes=None):
        volume = sitk.ReadImage(file)
        mask_volume = sitk.ReadImage(mask, outputPixelType=sitk.sitkUInt8)
        volume = torchio.CropOrPad(mask_volume.GetSize())(volume)
        mask_current_class = copy.deepcopy(mask)
        module_found = False
        class_found = False
        for name, module_ in inspect.getmembers(radiomics, inspect.ismodule):
            if radiomic.lower() in name:
                rad_module = module_
                module_found = True
        if not module_found:
            warnings.warn(f"No radiomic module named {radiomic.lower()}")
            return
        for name, class_ in inspect.getmembers(rad_module, inspect.isclass):
            if radiomic in name:
                rad_class = class_
                class_found = True
        if not class_found:  # Should never happen with pyradiomics in current (3.0.1) impl. if module check passes
            warnings.warn(f"No class named {radiomic}")
            return
        print("Finding available classes")
        available_classes = np.unique(sitk.GetArrayFromImage(mask_volume))
        print("Found available classes")
        fo_features_dictionnary = {}
        keys_done = False
        if indexes is not None:
            for i in indexes:
                print(f"Class {i}")
                if i not in available_classes:
                    warnings.warn(f"Class {i} was not found in the mask")
                else:
                    mask_current_class = mask_current_class[mask_current_class == i] * 1
                    print(f"Starting computations for {rad_class}")
                    rad_object = rad_class(volume, mask_volume)
                    rad_object._initCalculation()  # NOQA
                    print(f"Computations done")

                    # Introspection to fid all the class' relevant getters
                    for method_name, method in inspect.getmembers(rad_object, inspect.ismethod):
                        if method_name[0] == "_" or "get" not in method_name or "FeatureValue" not in method_name:
                            pass
                        else:
                            print(f"Retrieving {method_name}....", end="")
                            try:
                                fo_features_dictionnary[method_name.replace("FeatureValue", "").replace("get", "")] = \
                                    str(method()[0])
                            except (IndexError, TypeError, ValueError):
                                try:
                                    fo_features_dictionnary[
                                        method_name.replace("FeatureValue", "").replace("get", "")] = \
                                        str(method())
                                    print(f"\r{method_name} Done")
                                except (IndexError, TypeError, ValueError):
                                    fo_features_dictionnary[
                                        method_name.replace("FeatureValue", "").replace("get", "")] = \
                                        "nan"
                                    print(f"{method_name} Done, but returned nan\n")
                                    print(traceback.format_exc())
                            except DeprecationWarning:
                                pass
                            print("Done")
                            print()
                if not keys_done:
                    self.f.write(f",{','.join(fo_features_dictionnary.keys())}")
                    self.f.write("\n")
                    keys_done = True
                self.f.write(f"{i},{','.join([fo_features_dictionnary[ky] for ky in fo_features_dictionnary.keys()])}")
                self.f.write("\n")

    def localize_maxs(self, in_label_area: int = None, neighboring_mask=None, exclude=[], resample_target=None):

        use_neighbors = neighboring_mask is not None
        use_mask = in_label_area is not None

        for patient in sorted(self.patients_tr.keys()):
            print(f"[+] Processing patient {patient}")
            self.patients_max_coords[patient] = {}
            self.patients_min_coords[patient] = {}
            images = []
            for key in self.patients_tr[patient].keys():
                if key == "label":
                    if use_mask:
                        file = self.patients_tr[patient][key]
                        mask_nii = nib.load(file)
                else:
                    file = self.patients_tr[patient][key]
                    nii_curr = nib.load(file)
                    if key == resample_target:
                        if resample_target is not None:
                            tr = torchio.Resample(np.abs(np.diag(nii_curr.affine)[:3]))
                        else:
                            tr = None
                    images.append(nii_curr)
            mask_nii = tr(mask_nii)
            mask = np.array(mask_nii.dataobj)
            mask_aff = np.array(mask_nii.affine)
            try:
                bb = get_bb_from_mask(mask_nii)
            except NameError as e:
                raise NameError(f" {e.name} not defined. If this is mask_nii, this probably means no "
                                f"label was found for patient {patient}")

            for idx_image in range(len(images)):
                image = images[idx_image]
                image = tr(image)
                mod_name = self.channels[str(idx_image)]
                if mod_name in exclude:
                    pass
                else:
                    print(f"    [+] Handling modality: {mod_name}")
                    aff = image.affine
                    dx, dy, dz = get_spacing(aff)
                    x0, y0, z0 = get_origin(aff)
                    matrix = np.array(image.dataobj)
                    h, w, d = np.shape(matrix)
                    maximum = -np.inf
                    minimum = np.inf
                    coords_min = (np.nan, np.nan, np.nan)
                    coords_max = (np.nan, np.nan, np.nan)
                    bb_image_space = (int(floor((bb[0] - x0) / dx)), int(floor((bb[1] - y0) / dy)), int(floor((bb[2] - z0) / dz)),
                                      int(ceil((bb[3] - x0) / dx)), int(ceil((bb[4] - y0) / dy)), int(ceil((bb[5] - z0) / dz)))
                    """print(len(range(max(0, bb_image_space[0]), min(h, bb_image_space[3]))) *
                          len(range(max(0, bb_image_space[1]), min(w, bb_image_space[4]))) *
                          len(range(max(0, bb_image_space[2]), min(d, bb_image_space[5]))))
                    print(bb_image_space)
                    print(h, w, d)"""
                    for coord_x in range(max(0, bb_image_space[0]), min(h, bb_image_space[3])):
                        real_coord_x = x0 + coord_x * dx
                        for coord_y in range(max(0, bb_image_space[1]), min(w, bb_image_space[4])):
                            real_coord_y = y0 + coord_y * dy
                            for coord_z in range(max(0, bb_image_space[2]), min(d, bb_image_space[5])):
                                real_coord_z = z0 + coord_z * dz
                                if use_mask:
                                    if space_point_in_mask(mask, mask_aff, (real_coord_x, real_coord_y, real_coord_z),
                                                           margins=(0, 0, 0)):
                                        if use_neighbors:
                                            nm_x, nm_y, nm_z = np.shape(neighboring_mask)
                                            if np.sum(matrix[coord_x-nm_x//2:coord_x+nm_x//2 + 1,
                                                      coord_y-nm_y//2:coord_y+nm_y//2 + 1,
                                                      coord_z-nm_z//2:coord_z + nm_z//2 + 1] * neighboring_mask) > maximum:
                                                coords_max = (real_coord_x, real_coord_y, real_coord_z)
                                                maximum = matrix[coord_x, coord_y, coord_z]
                                            if np.sum(matrix[coord_x-nm_x//2:coord_x+nm_x//2 + 1,
                                                      coord_y-nm_y//2:coord_y+nm_y//2 + 1,
                                                      coord_z-nm_z//2:coord_z + nm_z//2 + 1] * neighboring_mask) < minimum:
                                                minimum = matrix[coord_x, coord_y, coord_z]
                                                coords_min = (real_coord_x, real_coord_y, real_coord_z)
                                        else:
                                            if matrix[coord_x, coord_y, coord_z] > maximum:
                                                coords_max = (real_coord_x, real_coord_y, real_coord_z)
                                                maximum = matrix[coord_x, coord_y, coord_z]
                                            if matrix[coord_x, coord_y, coord_z] < minimum:
                                                minimum = matrix[coord_x, coord_y, coord_z]
                                                coords_min = (real_coord_x, real_coord_y, real_coord_z)
                    self.patients_max_coords[patient][mod_name] = coords_max
                    self.patients_min_coords[patient][mod_name] = coords_min
        self.compute_distances(self.patients_max_coords, self.patients_max_coords, euclidean3d, "max_dist.csv", normalize=True, resample_transform=tr)
        self.compute_distances(self.patients_min_coords, self.patients_min_coords, euclidean3d, "min_dist.csv", normalize=True, resample_transform=tr)
        self.compute_distances(self.patients_max_coords, self.patients_min_coords, euclidean3d, "min_max_dist.csv", normalize=True, resample_transform=tr)
        pickle.dump(self.patients_max_coords, open("patients_max_coords.pkl", "wb"))
        pickle.dump(self.patients_min_coords, open("patients_min_coords.pkl", "wb"))

    def compute_distances(self, dictionary1, dictionary2, distance_func, output_csv_dist , normalize: bool = False,
                          resample_transform=None):
        for patient in dictionary1:
            print(f"Computing distances for {patient}")
            f = open(f"out_csvs/{patient}_{output_csv_dist}", "w")
            for mod1 in dictionary1[patient].keys():
                if normalize:
                    nii = nib.load(self.patients_tr[patient][mod1])
                    if resample_transform is not None:
                        nii = resample_transform(nii)
                    nb_vxls = np.unique(np.array(nii.dataobj), return_counts=True)[1][1]
                    size_voxel = np.prod(np.diag(nii.affine))  # assuming aff [4,4] == 1
                    vol = nb_vxls * size_voxel
                else:
                    vol = 1.
                f.write(mod1)
                for mod2 in dictionary2[patient].keys():
                    f.write(f",{distance_func(dictionary1[patient][mod1], dictionary2[patient][mod2])/vol}")
                f.write("\n")
            f.close()

    def _export_dict_to_slicer_markups(self, dictionary, f_ident: str):

        for patient in dictionary.keys():
            dict_ = {"@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/"
                                "Markups/Resources/Schema/markups-schema-v1.0.3.json#", "markups": [{}]}
            dict_["markups"][0]["type"] = "Fiducial"
            dict_["markups"][0]["coordinateSystem"] = "HFS"
            dict_["markups"][0]["coordinateUnits"] = "mm"
            dict_["markups"][0]["locked"] = True
            dict_["markups"][0]["fixedNumberOfControlPoints"] = True
            dict_["markups"][0]["controlPoints"] = []
            for idx, (modality, coords_) in enumerate(dictionary[patient].items()):
                dict_["markups"][0]["controlPoints"].append({})
                dict_["markups"][0]["controlPoints"][idx]["id"] = str(idx + 1)
                dict_["markups"][0]["controlPoints"][idx]["label"] = f"{modality}_{f_ident}"
                dict_["markups"][0]["controlPoints"][idx]["position"] = [coords_[0], coords_[1], coords_[2]]
            json.dump(dict_, open(f"markups/{f_ident}_{patient}.mrk.json", "w"))

    def export_coords_to_slicer_markups(self):
        """
        Exports the obtained max and min coordinates to 3D Slicer markups,
        so they can be visualized in this software
        :return: None
        """
        self._export_dict_to_slicer_markups(self.patients_max_coords, f_ident="max")
        self._export_dict_to_slicer_markups(self.patients_min_coords, f_ident="min")
        """for patient in self.patients_min_coords.keys():
            dict_min = {"@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/" \
                                   "Markups/Resources/Schema/markups-schema-v1.0.3.json#", "markups": [{}]}
            dict_min["markups"][0]["type"] = "Fiducial"
            dict_min["markups"][0]["coordinateSystem"] = "HFS"
            dict_min["markups"][0]["coordinateUnits"] = "mm"
            dict_min["markups"][0]["locked"] = True
            dict_min["markups"][0]["fixedNumberOfControlPoints"] = True
            dict_min["markups"][0]["controlPoints"] = []
            for idx, (modality, coords_min) in enumerate(self.patients_min_coords[patient].items()):
                dict_min["markups"][0]["controlPoints"].append({})
                dict_min["markups"][0]["controlPoints"][idx]["id"] = str(idx + 1)
                dict_min["markups"][0]["controlPoints"][idx]["label"] = f"{modality}_min"
                dict_min["markups"][0]["controlPoints"][idx]["position"] = [coords_min[0], coords_min[1], coords_min[2]]
            json.dump(dict_min, open(f"markups/min_{patient}.mrk.json", "w"))

        for patient in self.patients_max_coords.keys():
            dict_max = {"@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/" \
                                   "Markups/Resources/Schema/markups-schema-v1.0.3.json#", "markups": [{}]}
            dict_max["markups"][0]["type"] = "Fiducial"
            dict_max["markups"][0]["coordinateSystem"] = "HFS"
            dict_max["markups"][0]["coordinateUnits"] = "mm"
            dict_max["markups"][0]["locked"] = True
            dict_max["markups"][0]["fixedNumberOfControlPoints"] = True
            dict_max["markups"][0]["controlPoints"] = []
            for idx, (modality, coords_min) in enumerate(self.patients_max_coords[patient].items()):
                dict_max["markups"][0]["controlPoints"].append({})
                dict_max["markups"][0]["controlPoints"][idx]["id"] = str(idx + 1)
                dict_max["markups"][0]["controlPoints"][idx]["label"] = f"{modality}_max"
                dict_max["markups"][0]["controlPoints"][idx]["position"] = [coords_min[0], coords_min[1], coords_min[2]]
            json.dump(dict_max, open(f"markups/max_{patient}.mrk.json", "w"))"""


if __name__ == "__main__":
    msd_dataset = MSDDataset("/mnt/disk_2/Zach/Dataset508_RTEP8", "/mnt/disk_2/Zach/csvs_RTEP8")
    # msd_dataset.compute_radiomics(["PT_FMISO", "CT_FMISO"], ["FirstOrder"], selected_classes_indexes=[1])
#    mask_neighbors = np.array([[[0., 0., 0.], [0., .5, 0.], [0, 0, 0.]],
#              [[0., .5, 0.], [.5, 1., .5], [0., .5, 0.]],
#              [[0., 0., 0.], [0., .5, 0.], [0, 0, 0.]]])
    mask_neighbors = np.array([[[1/2*(np.sqrt(3)), 1/2*(np.sqrt(2)), 1/2*(np.sqrt(3))],
                                [1/2*(np.sqrt(2)), .5, 1/2*(np.sqrt(2))],
                                [1/2*(np.sqrt(3)), 1/2*(np.sqrt(2)), 1/2*(np.sqrt(3))]],
                               [[1/2*(np.sqrt(2)), .5, 1/2*(np.sqrt(2))],
                                [.5, 1., .5],
                                [1/2*(np.sqrt(2)), .5, 1/2*(np.sqrt(2))]],
                               [[1/2*(np.sqrt(3)), 1/2*(np.sqrt(2)), 1/2*(np.sqrt(3))],
                                [1/2*(np.sqrt(2)), .5, 1/2*(np.sqrt(2))],
                                [1/2*(np.sqrt(3)), 1/2*(np.sqrt(2)), 1/2*(np.sqrt(3))]]])
    mask_neighbors /= np.sum(mask_neighbors)
    msd_dataset.localize_maxs(in_label_area=1, exclude=["CT_FDG"], resample_target="PT_FMISO")
    msd_dataset.export_coords_to_slicer_markups()

# Sortir resampled abs / norm / with / without neighbors
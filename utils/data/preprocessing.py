import os
import nibabel as nib
import numpy as np
import torchio as tio


def get_median_spacing(dir_path):
    spacings_x = []
    spacings_y = []
    slice_thicknesses = []
    for file in dir_path:
        sx, sy, sz = nib.load(os.path.join(dir_path, file)).header.get_zooms()
        spacings_x.append(sx)
        spacings_y.append(sy)
        slice_thicknesses.append(sz)
    median_x = np.median(spacings_x)
    median_y = np.median(spacings_y)
    median_z = np.median(slice_thicknesses)
    return median_x, median_y, median_z



class MSDPreprocessor:
    def __init__(self, msd_path, dest_path, dest_spacing=None):
        self.msd_path = msd_path
        self.dest_path = dest_path
        self.patient_paths = []
        images_tr_path = os.path.join(self.msd_path, "imagesTr")
        images_ts_path = os.path.join(self.msd_path, "imagesTs")
        labels_tr_path = os.path.join(self.msd_path, "labelsTr")
        labels_ts_path = os.path.join(self.msd_path, "labelsTs")
        self.patients = []
        for file in os.listdir(labels_ts_path):
            self.patients.append(file.split(".")[0][-3:])
        for file in os.listdir(labels_tr_path):
            self.patients.append(file.split(".")[0][-3:])

        if dest_spacing is None:
            self.median_spacing = get_median_spacing(self.msd_path)
        self.resampling_transform = tio.Resample(self.median_spacing)

    def do_resampling_patient(self, patient_data, patient_mask: str):
        if type(patient_data) == str:
            patient_data = [patient_data]
        tio_args = {"name": patient_mask.replace(".nii.gz", "")}
        for data_idx in patient_data:
            tio_args[f"data_{data_idx}"] = tio.ScalarImage(patient_data[data_idx])
        tio_args[f"mask"] = tio.LabelMap(patient_mask)
        subject = tio.Subject(tio_args)
        subject = self.resampling_transform(subject)
        return subject, tio_args

    def save_subject_npy(self, subject):
        tio_args = subject.keys()
        name = tio_args.pop(["name"])
        tio_args.pop("mask")
        mask = subject["mask"]
        idx = 0
        for key in tio_args:
            idx += 1
            image = subject[key].numpy()
            save_name = name + str(idx).zfill(4) + ".npy"
            np.save(save_name, image)
        mask = mask.numpy()
        save_name_mask = name + ".npy"
        np.save(save_name_mask, mask)

    def process_patient(self, patient_number):
        for file in self.patient_paths:




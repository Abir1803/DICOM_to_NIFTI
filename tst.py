import copy
import datetime
import os

import pydicom
import torch
from matplotlib import pyplot as plt
from totalsegmentator.python_api import totalsegmentator
import numpy as np
import torchio
import nibabel as nib

ct_path = "/mnt/disk_2/Zach/RTEP7/CT"
tep_path = "/mnt/disk_2/Zach/RTEP7/PT"
output_path = "out_iguess/"
dcm_file_tep = os.listdir(tep_path)[0]
dcm_tep = pydicom.dcmread(os.path.join(tep_path, dcm_file_tep))
scantime = datetime.datetime.strptime(dcm_tep.AcquisitionTime, '%H%M%S')
injection_time = datetime.datetime.strptime(dcm_tep.RadiopharmaceuticalInformationSequence[0].
                                            RadiopharmaceuticalStartTime,
                                            '%H%M%S.%f')
half_life = float(dcm_tep.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
injected_dose = float(dcm_tep.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
decay = np.exp(-np.log(2) * ((scantime - injection_time).seconds) / half_life)
injected_dose_decay = injected_dose * decay
weight_grams = 75000
print(f"Scan Time: {scantime}")
print(f"Injection Time: {injection_time}")
print(f"Half Life: {half_life}")
print(f"Injected Dose: {injected_dose}")
print(f"Injected Dose Decay: {injected_dose_decay}")
suv = weight_grams / injected_dose_decay
print(suv)
if len(os.listdir(output_path)):
    pass
else:
    totalsegmentator(ct_path, output_path)

    res = None
    for count, nii_fpath in enumerate(sorted(os.listdir(output_path))):
        print(f"\r[+] {count}", end="")
        nii_file = nib.load(os.path.join(output_path, nii_fpath))
        aff = nii_file.affine
        np_arr = np.array(nii_file.dataobj)
        if res is None:
            res = np.zeros([105, *np_arr.shape], dtype=np.float32)
        res[count+1] = np_arr.astype(np.float32)

    print(res.shape)
    nib.save(nib.Nifti1Image(torch.argmax(torch.tensor(res), dim=0, keepdim=True).numpy()[0], affine=aff,
                                     dtype=np.float32), "out.nii.gz")
ct = torchio.ScalarImage(ct_path)
tep = torchio.ScalarImage(tep_path)
tep[torchio.DATA] = tep[torchio.DATA] * suv
organs = torchio.LabelMap("out.nii.gz")

sx, sy, sz = abs(ct[torchio.AFFINE][0, 0]), abs(ct[torchio.AFFINE][1, 1]), abs(ct[torchio.AFFINE][2, 2])
tr = torchio.Resample((sx, sy, sz))
tep = tr(tep)
ts = torchio.CropOrPad((ct.shape[1:]))
tep = ts(tep)
tc = torchio.ToCanonical()
organs = tc(organs)
tep = tc(tep)
ct = tc(ct)
subject = torchio.Subject(ct=ct, tep=tep, brain=organs)
print(subject["ct"][torchio.AFFINE])
print(subject["tep"][torchio.AFFINE])
print(subject["brain"][torchio.AFFINE])

voxel_volume = (sx * sy * sz)/1000  # in milliliters
brain_masked_tep = copy.deepcopy(tep.numpy())
brain_mask = organs.numpy() == 6  # brain
bladder_mask = organs.numpy() == 80  # urinary bladder
hear_al = organs.numpy() == 22  # heart atrium left
heart_ar = organs.numpy() == 23  # heart atrium left
heart_mask = organs.numpy() == 24  # heart myocardium
heart_vl = organs.numpy() == 25  # heart ventricle left
heart_vr = organs.numpy() == 26  # heart ventricle right
liver = organs.numpy() == 40  # liver
brain_masked_tep[brain_mask] = 0
brain_masked_tep[bladder_mask] = 0
brain_masked_tep[hear_al] = 0
brain_masked_tep[heart_ar] = 0
brain_masked_tep[heart_mask] = 0
brain_masked_tep[heart_vl] = 0
brain_masked_tep[heart_vr] = 0
brain_masked_tep[liver] = 0
print(np.max(subject['tep'][torchio.DATA][0].numpy()))
plt.subplot(2, 2, 1)
plt.imshow(np.mean(subject['ct'][torchio.DATA][0].numpy(), axis=1), "gray")
plt.subplot(2, 2, 2)
plt.imshow(np.mean(brain_mask[0], axis=1), "gray")

tep_mip = np.mean(subject['tep'][torchio.DATA][0].numpy(), axis=1)
max_ = np.max(tep_mip)
min_ = np.min(tep_mip)
plt.subplot(2, 2, 3)
plt.imshow(tep_mip, "gray", vmin=min_, vmax=max_)
plt.subplot(2, 2, 4)
plt.imshow(np.mean(brain_masked_tep[0], axis=1), "gray", vmin=min_, vmax=max_)
plt.tight_layout()
plt.savefig("mip_lung.png")

print(np.sum(brain_masked_tep) * voxel_volume)
print(voxel_volume)


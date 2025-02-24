import os

import nibabel
import numpy as np
from multiprocessing import spawn

import torch

root_dir = "out_iguess"

res = None
for count, nii_fpath in enumerate(sorted(os.listdir(root_dir))):
    print(f"\r[+] {count}", end="")
    nii_file = nibabel.load(os.path.join(root_dir, nii_fpath))
    aff = nii_file.affine
    np_arr = np.array(nii_file.dataobj)
    if res is None:
        res = np.zeros([105, *np_arr.shape], dtype=np.float32)
    res[count+1] = np_arr.astype(np.float32)

print(res.shape)
nibabel.save(nibabel.Nifti1Image(torch.argmax(torch.tensor(res), dim=0, keepdim=True).numpy()[0], affine=aff,
                                 dtype=np.float32), "out.nii.gz")
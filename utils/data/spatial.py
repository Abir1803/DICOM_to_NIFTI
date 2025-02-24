import os

import nibabel
import numpy as np
from matplotlib import pyplot as plt
from nibabel import Nifti1Image


def space_point_in_mask(mask, mask_affine, space_point, margins):
    x, y,  z = space_point
    mx, my, mz = margins
    dx, dy , dz = get_spacing(mask_affine)
    x0, y0, z0 = get_origin(mask_affine)
    if np.all([mx/dx < 0, my/dy < 0, mz/dz < 0]):
        negative_margin = True
        mx, my, mz = -mx, -my, -mz
    elif np.all([mx/dx >= 0, my/dy >= 0, mz/dz >= 0]):
        negative_margin = False
    else:
        raise NotImplementedError("All margins should be either positive or negative")
    x_discrete_min = round((x - mx - x0)/dx)
    y_discrete_min = round((y - my - y0)/dy)
    z_discrete_min = round((z - mz - z0)/dz)
    x_discrete_max = round((x + mx - x0)/dx)
    y_discrete_max = round((y + my - y0)/dy)
    z_discrete_max = round((z + mz - z0)/dz)
    try:
        if negative_margin:
            return np.min(mask[x_discrete_min:x_discrete_max + 1, y_discrete_min:y_discrete_max + 1,
                          z_discrete_min:z_discrete_max + 1])
        else:
            return np.max(mask[x_discrete_min:x_discrete_max+1, y_discrete_min:y_discrete_max+1, z_discrete_min:z_discrete_max+1])
    except IndexError:
        print(f"\rIndexError: {x_discrete_min}, {y_discrete_min}, {z_discrete_min}", end="")
        return False


def euclidean3d(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)


def get_origin(affine):
    return affine[0, 3], affine[1, 3], affine[2, 3]


def get_spacing(affine):
    return affine[0, 0], affine[1, 1], affine[2, 2]


def get_bb_from_mask(mask_nii: Nifti1Image):
    aff = mask_nii.affine
    dx, dy, dz = get_spacing(aff)
    x0, y0, z0 = get_origin(aff)
    wheres = np.argwhere(np.array(mask_nii.dataobj))
    min_x, min_y, min_z = np.min(wheres, axis=0)
    max_x, max_y, max_z = np.max(wheres, axis=0)
    return (x0 + dx * min_x, y0 + dy * min_y, z0 + dz * min_z,
            x0 + dx * max_x + 1, y0 + dy * max_y + 1, z0 + dz * max_z + 1)


if __name__ == "__main__":
    root_dir = "/mnt/disk_2/Zach/Dataset508_RTEP8"
    for file in os.listdir(os.path.join(root_dir, "imagesTr"))[ :1]:
        print(file)
        mask_nii = nibabel.load(os.path.join(root_dir, "labelsTr", file.split(".")[0][:-5]+".nii.gz"))
        mask = np.array(mask_nii.dataobj)
        t1_nii = nibabel.load(os.path.join(root_dir, "imagesTr", file))
        t1 = np.array(t1_nii.dataobj)
        aff = t1_nii.affine
        dx, dy, dz = get_spacing(aff)
        x0, y0, z0 = get_origin(aff)
        mask_aff = mask_nii.affine
        h, w, d = np.shape(t1)
        print(h, w, d)
        for coord_x in range(h):
            real_coord_x = x0 + coord_x * dx
            for coord_y in range(w):
                real_coord_y = y0 + coord_y * dy
                for coord_z in range(d):
                    real_coord_z = z0 + coord_z * dz
                    if not space_point_in_mask(mask, mask_aff, (real_coord_x, real_coord_y, real_coord_z)):
                        t1[coord_x, coord_y, coord_z] = 0.
        plt.subplot(1, 2, 1)
        plt.imshow(np.max(t1, axis=1))
        plt.subplot(1, 2, 2)
        plt.imshow(np.max(mask, axis=1))
        plt.savefig("test.png", dpi=1000)
        print(np.unique(t1))


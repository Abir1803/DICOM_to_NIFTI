import os
import shutil

import numpy as np


def maybe_mkdir(folder_path):
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        pass


def maybe_copy(src_path, dest_path):
    if os.path.exists(dest_path):
        return
    else:
        shutil.copy(src_path, dest_path)


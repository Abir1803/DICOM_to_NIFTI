# FLAP_Segmentation

This repository contains a Python script designed to convert medical imaging data from DICOM format to NIfTI format. This conversion facilitates easier handling and analysis of medical images in various research and clinical applications.

## Features

- **Batch Conversion**: Process all DICOM files within a specified directory.
- **Output Compression**: Save NIfTI files in compressed `.nii.gz` format to save disk space.
- **Error Handling**: Gracefully handles exceptions during the conversion process, ensuring robustness.

## Prerequisites

Before using the script, ensure that the following Python libraries are installed:

- `pydicom`: For reading DICOM files.
- `nibabel`: For creating NIfTI files.
- `numpy`: For numerical operations.

You can install these dependencies using pip:

```bash
pip install pydicom nibabel numpy
```

## Usage

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Abir1803/DICOM_to_NIFTI.git
   cd DICOM_to_NIFTI
   ```

2. **Run the Conversion Script**:

   ```bash
   python main.py msd
   ```

## Script Overview

The script performs the following steps:

1. **Load DICOM Files**: Reads all DICOM files from the specified input directory.
2. **Extract Image Data and Metadata**: Retrieves pixel data and relevant metadata from the DICOM files.
3. **Construct NIfTI Image**: Uses the extracted data to create a NIfTI image object.
4. **Save NIfTI File**: Writes the NIfTI image to the specified output directory in compressed `.nii.gz` format.

## References

- [pydicom Documentation](https://pydicom.github.io/)
- [nibabel Documentation](https://nipy.org/nibabel/)
- [NumPy Documentation](https://numpy.org/doc/)

For further information and advanced usage, please refer to the script [`main.py`](https://github.com/Abir1803/DICOM_to_NIFTI/blob/main/main.py) in this repository.

---

*Note: Ensure that your DICOM files are organized appropriately, and the output directory exists before running the script.*
"""

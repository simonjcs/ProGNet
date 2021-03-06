# Deep Learning Improves Speed and Accuracy of Prostate Gland Segmentations on MRI for Targeted Biopsy

This page aims to describe and share the code presented in the following paper:

Soerensen SJC, Fan RE, Seetharaman A, et al. "Deep Learning Improves Speed and Accuracy of Prostate Gland Segmentations on MRI for Targeted Biopsy". J Urol. 2021 Apr 21: [https://www.auajournals.org/doi/10.1097/JU.0000000000001783]

## Introduction
Accurate prostate segmentation on MRI is critical for biopsy, yet manual segmentation is tedious and time-consuming. To address this, we developed a deep learning model and used it in a clinical setting to rapidly and accurately segment the prostate on MRI and shared the code online.

The use of this code is for research purposes only. If you have any questions regarding how to use this code, please send an email to simonjcs [at] stanford [dot] edu. The code has been tested successfully on a Windows 10 machine with an NVIDIA V100 graphics card, Python 3.7, and Tensorflow 2.0.

## Citation

If you use this code in an academic paper, please cite our article:

```bibtex
@article{soerensen2021deep,
  title={Deep Learning Improves Speed and Accuracy of Prostate Gland Segmentations on MRI for Targeted Biopsy},
  author={Soerensen, Simon John Christoph and Fan, Richard E and Seetharaman, Arun and Chen, Leo and Shao, Wei and Bhattacharya, Indrani and Kim, Yong-hun and Sood, Rewa and Borre, Michael and Chung, Benjamin I and Sonn, Geoffrey A and Rusu, Mirabela},
  journal={The Journal of Urology},
  pages={10--1097},
  year={2021},
  publisher={Wolters Kluwer Philadelphia, PA}
}
```

 \* If you use the normalization step and/or use the U-net architecture, please cite the authors: https://github.com/jcreinhold/intensity-normalization & https://github.com/zhixuhao/unet

## Usage

#### 1. Install the dependencies:
- Python 3.7.6
- tensorflow 2.0.0
- numpy  1.18.4
- pandas 0.25.3
- pydicom 1.4.1
- SimpleITK 2.02
- matplotlib 3.1.1
- nibabel 3.0.1
- scipy 1.3.2
- pydicom-seg 0.1.0
- pyntcloud 0.1.2
- scikit-image 0.16.2
- scikit-learn 0.22.1
- trimesh
- vtk

#### 2. Download three files from this repository:

- [ProGNet_Segmentation.py](https://github.com/simonjcs/ProGNet/blob/main/ProGNet_Segmentation.py) (main code)
- [std_hist_T2.npy](https://github.com/simonjcs/ProGNet/blob/main/std_hist_T2.npy) (MRI intensities learned prior to model training)
- [prognet_t2.h5](https://github.com/simonjcs/ProGNet/blob/main/prognet_t2.h5) (model weights)

#### 3. Download the following code and cite the authors:

- https://github.com/jcreinhold/intensity-normalization/blob/master/intensity_normalization/normalize/nyul.py

#### 4. Download the following code and cite the authors:

- https://github.com/zhixuhao/unet/blob/master/model.py
 
#### 5. Prepare the MRI intensity normalization code for use:

- Put the nyul.py code in line 392 line of the main ProGNet_Segmentation.py after "def Step3(outputDir, standardHist): ## Place intensity normalization code here" and update the code for use.

#### 6. Prepare the deep learning model for use:

- Put the the model.py code in line 331 of the main ProGNet_Segmentation.py file after "## Place deep learning code here" and update this part for use.

#### 7. Update filepaths in the main ProGNet_Segmentation.py file:

- Update the input path in line 1277. This folder should contain T2-DICOM folders. 
- Update the output path in line 1278. This folder will (if the code was succesfully run) contain T2-DICOM folders that also include a SEG-DICOM file.
- Update the filepath in line 1279 to point to the downloaded std_hist_T2.npy file.
- Update the filepath in line 1280 to point to the progNet_t2.h5 model weights

#### 8. Run the code!

- Run the ProGNet_Segmentation.py file to run the ProGNet prostate clinical segmentation pipeline.

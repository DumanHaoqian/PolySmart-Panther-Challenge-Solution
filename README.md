# PolySmart-Panther-Challenge-Solution
*🎉 We got the 4th Place of Participation and be selected in Honorable Mention*<br>
*Our Checkpoints* **[Task1](https://drive.google.com/drive/folders/1HypwAE4xHDwy762LLGRCYTSBcYfADJRA?usp=sharing) & [Task2](https://drive.google.com/drive/folders/1HypwAE4xHDwy762LLGRCYTSBcYfADJRA?usp=sharing)**

## Task1: Random Deformation Augmented ResEncUNet with Dice Loss only and 3 folds Ensambale(PC-task1-Final)
### Random Deformation

The random deformation module applies smooth deformation fields to medical images and labels, ensuring realistic and anatomically plausible augmentations for deep learning training. A smooth deformation field **D(x)** is generated using a combination of random displacements, Gaussian smoothing, and distance-based weighting. For each labeled structure, the displacement **d(x)** at a voxel **x** is calculated as:

**d(x) = w(x) * [S(x) * (1 + r_s) + r_d]**

where **w(x)** is a distance-based weight (e.g., **w(x) = exp(-||x - c|| / λ)**, with **c** as the structure's center of mass), **S(x)** is the relative position of **x** to **c**, **r_s** is a random scaling factor, and **r_d** is a random global displacement. The resulting field is smoothed using multi-scale Gaussian filters to avoid abrupt changes. The deformation field is then applied to the image or label using a displacement field transform **T(x) = x + D(x)**. Key constraints include limiting the maximum displacement **||D(x)|| ≤ d_max** to prevent artifacts and ensuring the field is free of NaN or Inf values. The deformed output is interpolated using linear or nearest-neighbor methods (for images and labels, respectively) and saved in NIfTI format. This approach generates realistic spatial variability while preserving structural integrity, making it ideal for biomedical image segmentation tasks.

## Task2: Ultra-Random Deformation Augmented TotalSegmentator-Based ROI ResEncUNet with Dice Loss only and 3 folds Ensambale(原神，启动！)


## Leadearboard 
**We are at the 5th place: PC-task1-Final**
**![Leader Board of Task1](https://github.com/DumanHaoqian/PolySmart-Panther-Challenge-Solution/blob/main/Images/LB1.png)**<br>
**We are at the 5th place: 原神，启动！(＾Ｕ＾)ノ~**
**![Leader Board of Task2](https://github.com/DumanHaoqian/PolySmart-Panther-Challenge-Solution/blob/main/Images/LB2.png)**<br>

## References

**This project builds upon the following repositories:**

**1. [PANTHER_baseline](https://github.com/DIAGNijmegen/PANTHER_baseline)  
   Licensed under [Apache 2.0 License](https://github.com/MIC-DKFZ/nnUNet/blob/master/LICENSE).  
   Original repository focuses on baseline methods for pancreas tumor segmentation.**

**2. [nnUNet](https://github.com/MIC-DKFZ/nnUNet)  
   Licensed under [Apache 2.0 License](https://github.com/MIC-DKFZ/nnUNet/blob/master/LICENSE).  
   nnUNet is a self-adapting framework for biomedical image segmentation.**

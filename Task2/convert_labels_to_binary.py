import SimpleITK as sitk
import numpy as np
from pathlib import Path


def convert_labels_to_binary(labels_dir):
    """
    将标签文件中的类别2转换为类别0，保持0,1,2 -> 0,1,0
    """
    labels_path = Path(labels_dir)

    for label_file in labels_path.glob("*.nii.gz"):
        # 读取标签
        label_img = sitk.ReadImage(str(label_file))
        label_array = sitk.GetArrayFromImage(label_img)

        # 检查唯一值
        unique_values = np.unique(label_array)
        print(f"{label_file.name}: 唯一值 {unique_values}")

        # 如果包含类别2，转换为0
        if 2 in unique_values:
            label_array[label_array == 2] = 0

            # 保存修改后的标签
            new_label_img = sitk.GetImageFromArray(label_array)
            new_label_img.CopyInformation(label_img)
            sitk.WriteImage(new_label_img, str(label_file))

            print(f"已转换 {label_file.name}: 类别2 -> 类别0")


# 使用
convert_labels_to_binary(r"D:\t2_pancreas_project\data\processed\temp_segmentations\labeled_labels")
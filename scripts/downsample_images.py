from pathlib import Path
import SimpleITK as sitk

import SimpleITK as sitk
import nibabel as nib
import numpy as np
from pathlib import Path


def batch_resample(input_dir, output_dir):
    """
    重采样图像并自动转换为RAS方向
    Args:
        input_dir: 输入目录（包含.nii.gz文件）
        output_dir: 输出目录
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    #for img_file in input_path.glob("*.nii.gz"):
    for img_file in input_path.glob("*.mha"):
        try:
            # 使用SimpleITK进行重采样
            img = sitk.ReadImage(str(img_file))
            original_spacing = img.GetSpacing()
            target_spacing = (3, 3, 6)

            new_size = [
                int(img.GetSize()[i] * original_spacing[i] / target_spacing[i])
                for i in range(3)
            ]

            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(target_spacing)
            resampler.SetSize(new_size)
            resampler.SetOutputOrigin(img.GetOrigin())
            resampler.SetOutputDirection(img.GetDirection())
            resampler.SetInterpolator(sitk.sitkLinear)

            resampled = resampler.Execute(img)

            # 临时保存重采样结果
            temp_file = output_path / f"temp_{img_file.name}"
            sitk.WriteImage(resampled, str(temp_file))

            # 使用nibabel转换为RAS方向
            nib_img = nib.load(str(temp_file))
            ras_img = nib.as_closest_canonical(nib_img)

            # 保存最终结果
            output_file = output_path / img_file.name
            nib.save(ras_img, str(output_file))
            #nib.save(nib_img,str(output_file))
            # 删除临时文件
            temp_file.unlink()

            print(f"完成并转换为RAS方向: {img_file.name}")

        except Exception as e:
            print(f"处理失败 {img_file.name}: {str(e)}")
            if 'temp_file' in locals() and temp_file.exists():
                temp_file.unlink()  # 确保删除临时文件




# 使用
if __name__ == "__main__":
    batch_resample("D:\PANTHER_baseline-main\PANTHER_baseline-main\TASK2_baseline\input\images", "D:\PANTHER_baseline-main\PANTHER_baseline-main\TASK2_baseline\input")
    #batch_resample("../data/raw_labeled/ImagesTr", "D:/t2_pancreas_project/data/processed/temp_downsampled/labeled")

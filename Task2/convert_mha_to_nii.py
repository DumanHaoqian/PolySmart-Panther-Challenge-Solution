import SimpleITK as sitk
from pathlib import Path


def convert_mha_to_nii(input_dir):
    input_path = Path(input_dir)

    for mha_file in input_path.glob("*.mha"):
        # 读取并转换
        img = sitk.ReadImage(str(mha_file))
        nii_file = mha_file.with_suffix('.nii.gz')
        sitk.WriteImage(img, str(nii_file))

        # 删除原文件
        mha_file.unlink()
        print(f"转换完成: {mha_file.name} -> {nii_file.name}")


# 使用
convert_mha_to_nii("./data/raw_labeled/ImagesTr")
convert_mha_to_nii("./data/raw_labeled/LabelsTr")
#"C:\Users\lenovo\Downloads\PANTHER_Task1\LabelsTr"
convert_mha_to_nii("C:/Users\lenovo\Downloads\PANTHER_Task1\LabelsTr")
convert_mha_to_nii("C:/Users\lenovo\Downloads\PANTHER_Task1\ImagesTr")


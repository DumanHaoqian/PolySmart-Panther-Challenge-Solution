#!/usr/bin/env python3
"""
nnU-Net文件重命名和移动脚本
"""

import os
import shutil
import glob
from pathlib import Path
import json


def create_nnunet_structure():
    """创建nnU-Net目录结构"""
    base_dir = Path("D:/t2_pancreas_project/nnUNet_raw/nnUNet_raw_data/Dataset005_PancreasT2")

    (base_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (base_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

    return base_dir


def rename_and_move_files():
    """重命名和移动文件"""

    source_paths = {
        "labeled_images": "D:/t2_pancreas_project/data/processed/temp_segmentations/labeled_rois",
        "labeled_labels": "D:/t2_pancreas_project/data/processed/temp_segmentations/labeled_labels",
        #"sgdef_images": "D:/t2_pancreas_project/data/processed/sgDefAug/ImagesTr",
        #"sgdef_labels": "D:/t2_pancreas_project/data/processed/sgDefAug/LabelsTr"
    }

    dataset_dir = create_nnunet_structure()
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"

    counter = 1

    # 处理labeled数据
    print("=== 处理labeled数据 ===")
    labeled_images = glob.glob(os.path.join(source_paths["labeled_images"], "*.nii.gz"))
    print(f"找到labeled图像文件: {len(labeled_images)}个")

    for img_path in labeled_images:
        img_file = Path(img_path)
        base_name = img_file.stem.replace('.nii', '')

        label_file = Path(source_paths["labeled_labels"]) / f"{base_name}.nii.gz"

        if label_file.exists():
            new_case_id = f"pancreas_{counter:04d}"

            new_img_name = f"{new_case_id}_0000.nii.gz"
            shutil.copy2(img_path, images_dir / new_img_name)

            new_label_name = f"{new_case_id}.nii.gz"
            shutil.copy2(label_file, labels_dir / new_label_name)

            print(f"✓ labeled: {base_name} -> {new_case_id}")
            counter += 1
        else:
            print(f"✗ 缺少labeled标签: {base_name}")

    print(f"labeled处理完成，当前数量: {counter - 1}")

    # 处理sgDefAug数据
    print("=== 处理sgDefAug数据 ===")
    sgdef_images = glob.glob(os.path.join(source_paths["sgdef_images"], "*.nii.gz"))
    print(f"找到sgDefAug图像文件: {len(sgdef_images)}个")

    sgdef_processed = 0
    sgdef_failed = 0

    for img_path in sgdef_images:
        img_file = Path(img_path)
        img_name = img_file.name  # 例如: 10303_0000_0001.nii.gz
        base_name = img_file.stem.replace('.nii', '')  # 10303_0000_0001

        # 直接查找同名标签文件
        label_file = Path(source_paths["sgdef_labels"]) / f"{base_name}.nii.gz"

        if label_file.exists():
            new_case_id = f"pancreas_{counter:04d}"

            # 复制图像
            new_img_name = f"{new_case_id}_0000.nii.gz"
            shutil.copy2(img_path, images_dir / new_img_name)

            # 复制标签
            new_label_name = f"{new_case_id}.nii.gz"
            shutil.copy2(label_file, labels_dir / new_label_name)

            print(f"✓ sgdef: {img_name} -> {new_case_id}")
            counter += 1
            sgdef_processed += 1
        else:
            print(f"✗ 缺少sgdef标签: {img_name}")
            sgdef_failed += 1

    print(f"sgDefAug处理完成: 成功{sgdef_processed}个, 失败{sgdef_failed}个")
    print(f"总处理数量: {counter - 1}")

    return counter - 1


def create_dataset_json(num_training):
    """创建dataset.json"""
    dataset_dir = Path("D:/t2_pancreas_project/nnUNet_raw/nnUNet_raw_data/Dataset005_PancreasT2")

    dataset_json = {
        "channel_names": {
            "0": "T2"
        },
        "labels": {
            "background": 0,
            "tumor": 1
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz"
    }

    with open(dataset_dir / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"已创建dataset.json，包含{num_training}个训练样本")


def verify_files():
    """验证结果"""
    dataset_dir = Path("D:/t2_pancreas_project/nnUNet_raw/nnUNet_raw_data/Dataset005_PancreasT2")
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"

    images = sorted(list(images_dir.glob("*.nii.gz")))
    labels = sorted(list(labels_dir.glob("*.nii.gz")))

    print(f"\n=== 验证结果 ===")
    print(f"图像文件数量: {len(images)}")
    print(f"标签文件数量: {len(labels)}")

    # 显示前几个文件示例
    print(f"\n图像文件示例:")
    for img in images[:3]:
        print(f"  {img.name}")

    print(f"\n标签文件示例:")
    for label in labels[:3]:
        print(f"  {label.name}")

    # 检查配对
    mismatched = []
    for img in images:
        case_id = img.name.replace('_0000.nii.gz', '')  # pancreas_0001
        expected_label = labels_dir / f"{case_id}.nii.gz"
        if not expected_label.exists():
            mismatched.append(case_id)

    if mismatched:
        print(f"\n警告: 以下样本缺少配对标签: {mismatched}")
    else:
        print(f"\n✓ 所有图像都有对应标签文件")


def main():
    print("开始nnU-Net文件重命名和移动...")

    # 检查源目录
    source_dirs = [
        "D:/t2_pancreas_project/data/processed/temp_segmentations/labeled_rois",
        "D:/t2_pancreas_project/data/processed/temp_segmentations/labeled_labels",
        "D:/t2_pancreas_project/data/processed/sgDefAug/ImagesTr",
        "D:/t2_pancreas_project/data/processed/sgDefAug/LabelsTr"
    ]

    for dir_path in source_dirs:
        if not os.path.exists(dir_path):
            print(f"错误: 源目录不存在 {dir_path}")
            return
        else:
            file_count = len(glob.glob(os.path.join(dir_path, "*.nii.gz")))
            print(f"发现源目录: {dir_path} ({file_count}个文件)")

    num_training = rename_and_move_files()

    if num_training > 0:
        create_dataset_json(num_training)
        verify_files()

        print(f"\n=== 完成! ===")
        print(f"总共处理了 {num_training} 个训练样本")
        print(f"目标目录: D:/t2_pancreas_project/nnUNet_raw/nnUNet_raw_data/Dataset005_PancreasT2")
        print(f"\n下一步运行:")
        print(f"nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity")
    else:
        print("未找到有效的文件对，请检查源目录和文件命名")


if __name__ == "__main__":
    #main()
    create_dataset_json(num_training=50)

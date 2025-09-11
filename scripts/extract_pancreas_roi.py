#!/usr/bin/env python3
"""
TotalSegmentator胰腺mask提取和ROI处理器
从TotalSegmentator输出中提取胰腺mask，重命名并生成ROI
"""

import os
import subprocess
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import argparse
import shutil
import logging
import nibabel as nib


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PancreasROIProcessor:
    #def __init__(self, labeled_dir, unlabeled_dir, output_dir, roi_margin=30):
    def __init__(self, labeled_dir, output_dir, roi_margin=30):
        """
        初始化处理器
        Args:
            labeled_dir: labeled数据目录
            unlabeled_dir: unlabeled数据目录
            output_dir: 输出目录
            roi_margin: ROI边距(mm)
        """

        self.labeled_dir = Path(labeled_dir)
        #self.unlabeled_dir = Path(unlabeled_dir)
        self.output_dir = Path(output_dir)
        self.roi_margin = roi_margin

        # 原始分辨率图像路径
        self.original_labeled_dir = Path("/data/raw_labeled/ImagesTr")
        self.original_labeled_labels_dir = Path("/data/raw_labeled/LabelsTr")
        #self.original_unlabeled_dir = Path("D:/t2_pancreas_project/data/raw_unlabeled/ImagesTr")
        #self.original_unlabeled_labels_dir = Path("D:/t2_pancreas_project/data/raw_unlabeled/LabelsTr")

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labeled_masks").mkdir(exist_ok=True) #totalsegmentator分割结果
        #(self.output_dir / "unlabeled_masks").mkdir(exist_ok=True)
        (self.output_dir / "labeled_rois").mkdir(exist_ok=True) #原始图像根据totalsegmentator分割结果切割出来的、原分辨率的roi
        #(self.output_dir / "unlabeled_rois").mkdir(exist_ok=True)
        (self.output_dir / "labeled_labels").mkdir(exist_ok=True) #根据roi切割出来的、对应的label
        #(self.output_dir / "unlabeled_labels").mkdir(exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            logger.error(f"无写入权限: {output_dir}")
    def get_nii_files(self, directory):
        """获取目录下所有.nii.gz文件的绝对路径"""
        if not directory.exists():
            logger.warning(f"目录不存在: {directory}")
            return []

        nii_files = list(directory.glob("*.nii.gz"))
        nii_files = [f.resolve() for f in nii_files]  # 转换为绝对路径

        logger.info(f"在 {directory} 找到 {len(nii_files)} 个.nii.gz文件")
        return nii_files

    def find_original_image(self, downsampled_file, data_type):
        """查找对应的原始分辨率图像"""
        #if data_type == "labeled":
        original_image_dir = self.original_labeled_dir
        original_label_dir = self.original_labeled_labels_dir
        #else:
            #original_image_dir = self.original_unlabeled_dir
            #original_label_dir = self.original_unlabeled_labels_dir

        # 提取文件ID
        file_id = downsampled_file.name
        if file_id.endswith('.nii.gz'):
            file_id = file_id[:-7]
        elif file_id.endswith('.nii'):
            file_id = file_id[:-4]

        # 移除前缀

        # 尝试匹配原始文件
        image_patterns = [
            f"{file_id}.nii.gz",
            f"{file_id}_0000.nii.gz",
            f"{file_id}_0001.nii.gz"
        ]
        original_image_path = None
        for pattern in image_patterns:
            # 查找图像
            if original_image_path is None:
                image_path = original_image_dir / pattern
                if image_path.exists():
                    original_image_path = image_path
                    logger.info(f"找到原始图像: {image_path.name}")


        label_patterns = [
            f"{file_id}.nii.gz",  # 直接匹配标签名（如10303.nii.gz），default，符合nnunet命名规则
            f"{file_id.split('_0000')[0]}.nii.gz",  # 如果file_id含_0000则去掉
            f"{file_id.split('_0002')[0]}.nii.gz" # for unlabeled dataset
        ]
        original_label_path = None
        for pattern in label_patterns:
            label_path = original_label_dir / pattern
            if label_path.exists():
                original_label_path = label_path
                logger.info(f"找到原始标签: {label_path.name}")
                break

        if original_image_path is None:
            logger.warning(f"未找到原始图像: {file_id}")
        if original_label_path is None:
            logger.warning(f"未找到原始标签: {file_id}")

        return original_image_path, original_label_path

    def run_totalsegmentator(self, input_file, output_dir):
        """运行TotalSegmentator分割"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "TotalSegmentator",
                "-i", str(input_file),
                "-o", str(output_dir),
                "-ta", "total_mr",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"TotalSegmentator分割完成: {input_file.name}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"TotalSegmentator分割失败 {input_file.name}: {e}")
            return False


    def crop_roi_from_mask(self, original_image_path, original_label_path, pancreas_mask_path, output_subdir):
        """根据胰腺mask在原分辨率上裁剪ROI"""
        # 加载原始分辨率图像
        original_img = sitk.ReadImage(str(original_image_path))
        # 加载降采样后的mask
        pancreas_mask = sitk.ReadImage(str(pancreas_mask_path))
        original_label = sitk.ReadImage(str(original_label_path))
        mask_array = sitk.GetArrayFromImage(pancreas_mask)
        if mask_array.sum() == 0:
            logger.warning(f"胰腺mask为空: {pancreas_mask_path.name}")
            return None

        # 将mask重采样到原始分辨率
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(original_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        pancreas_mask_original = resampler.Execute(pancreas_mask)
        mask_array_original = sitk.GetArrayFromImage(pancreas_mask_original)

        # 在原始分辨率上找到胰腺区域
        coords = np.where(mask_array_original > 0)
        if len(coords[0]) == 0:
            logger.warning(f"重采样后mask为空: {pancreas_mask_path.name}")
            return None

        min_z, max_z = int(coords[0].min()), int(coords[0].max())
        min_y, max_y = int(coords[1].min()), int(coords[1].max())
        min_x, max_x = int(coords[2].min()), int(coords[2].max())

        # 根据原始分辨率计算边距
        spacing = original_img.GetSpacing()
        margin_x = int(self.roi_margin / spacing[0])
        margin_y = int(self.roi_margin / spacing[1])
        margin_z = int(self.roi_margin / spacing[2])

        # 图像边界
        size_x, size_y, size_z = original_img.GetSize()

        # 安全边界
        start_x = max(0, min_x - margin_x)
        start_y = max(0, min_y - margin_y)
        start_z = max(0, min_z - margin_z)

        end_x = min(size_x, max_x + margin_x + 1)
        end_y = min(size_y, max_y + margin_y + 1)
        end_z = min(size_z, max_z + margin_z + 1)

        # ROI参数
        roi_index = [start_x, start_y, start_z]
        roi_size = [end_x - start_x, end_y - start_y, end_z - start_z]

        try:
            # 在原始分辨率图像上提取ROI
            extractor = sitk.ExtractImageFilter()
            extractor.SetSize(roi_size)
            extractor.SetIndex(roi_index)
            roi_img = extractor.Execute(original_img)
            roi_label = None
            if original_label is not None:
                roi_label = extractor.Execute(original_label)

            original_name = Path(original_image_path).name  # 例如 "10303_0000.nii.gz"
            clean_stem = original_name.replace(".nii.gz", "")  # 去掉后缀 → "10303_0000"

            # 构建输出文件名（保留_0000后缀）
            roi_output_path = self.output_dir / f"{output_subdir.replace('_masks', '_rois')}" / f"{clean_stem}.nii.gz"
            sitk.WriteImage(roi_img, str(roi_output_path))

            logger.info(f"✓ 原分辨率ROI已保存: {roi_output_path.name}")

            label_output_path = None
            if roi_label is not None:
                label_output_path = self.output_dir / f"{output_subdir.replace('_masks', '_labels')}" / f"{clean_stem}.nii.gz"
                sitk.WriteImage(roi_label, str(label_output_path))
                logger.info(f"✓ 标签ROI已保存: {label_output_path.name}")

            return roi_output_path, label_output_path

        except Exception as e:
            logger.error(f"ROI提取失败: {e}")
            return None

    def extract_and_rename_pancreas(self, output_dir, original_file, output_subdir):
        """
        从TotalSegmentator输出中提取胰脏mask并重命名

        Args:
            seg_output_dir: TotalSegmentator输出的目录
            original_file: 原始输入文件路径
            output_subdir: 输出子目录（labeled_masks/unlabeled_masks）

        Returns:
            Path: 重命名后的胰脏mask路径，如果失败返回None
        """
        try:
            # 1. 确定胰脏mask文件路径
            # TotalSegmentator输出的胰脏mask通常命名为"pancreas.nii.gz"
            pancreas_mask_path = output_dir / "pancreas.nii.gz"

            # 如果不存在，尝试其他可能的命名（不同版本可能不同）
            if not pancreas_mask_path.exists():
                pancreas_mask_path = output_dir / "segmentations" / "pancreas.nii.gz"

            if not pancreas_mask_path.exists():
                logger.error(f"未找到胰脏mask文件: {output_dir}")
                return None

            # 2. 生成目标文件名（与原始文件同名）
            original_stem = original_file.stem
            if original_stem.startswith("down_"):
                original_stem = original_stem[5:]  # 移除"down_"前缀

            if "_0000" in original_stem:
                original_stem = original_stem.split("_0000")[0]

            target_filename = f"{original_stem}.nii.gz"

            # 3. 创建输出目录（如果不存在）
            output_dir = self.output_dir / output_subdir
            output_dir.mkdir(parents=True, exist_ok=True)

            # 4. 复制并重命名文件
            target_path = output_dir / target_filename
            shutil.copy2(str(pancreas_mask_path), str(target_path))

            logger.info(f"胰脏mask已提取并重命名为: {target_filename}")
            return target_path

        except Exception as e:
            logger.error(f"提取胰脏mask失败: {e}")
            return None
    def process_file(self, img_file, output_subdir, file_index, total_files, data_type):
        """处理单个文件"""
        try:
            logger.info(f"[{file_index}/{total_files}] 处理: {img_file.name}")

            # 查找对应的原始分辨率图像
            original_image_path, original_label_path = self.find_original_image(img_file, data_type)
            if original_image_path is None:
                logger.error(f"未找到原始图像，跳过: {img_file.name}")
                return False

            # 1. TotalSegmentator分割（使用降采样图像）
            seg_output_dir = self.output_dir / f"{img_file.stem}"
            if not self.run_totalsegmentator(img_file, seg_output_dir):
                return False

            # 2. 提取并重命名胰腺mask
            pancreas_mask_path = self.extract_and_rename_pancreas(
                seg_output_dir, img_file, output_subdir
            )
            if pancreas_mask_path is None:
                return False

            # 3. 在原始分辨率上生成ROI
            #file_id = pancreas_mask_path.stem
            #self.crop_roi_from_mask(original_image_path,original_label_path, pancreas_mask_path, file_id, output_subdir)
            self.crop_roi_from_mask(original_image_path, original_label_path, pancreas_mask_path,output_subdir)

            # 4. 清理临时分割目录
            shutil.rmtree(seg_output_dir, ignore_errors=True)

            return True

        except Exception as e:
            logger.error(f"处理失败 {img_file.name}: {e}")
            return False

    def process_dataset(self, data_dir, output_subdir, data_type):
        """处理数据集"""
        logger.info(f"处理{data_type}数据: {data_dir}")

        nii_files = self.get_nii_files(data_dir)
        if not nii_files:
            return

        success_count = 0
        for i, img_file in enumerate(nii_files, 1):
            if self.process_file(img_file, output_subdir, i, len(nii_files), data_type):
                success_count += 1

        logger.info(f"{data_type}处理完成: {success_count}/{len(nii_files)} 成功")

    def run(self):
        """运行完整处理流程"""
        logger.info("开始胰腺mask提取和原分辨率ROI处理")

        # 检查TotalSegmentator
        try:
            subprocess.run([r"TotalSegmentator", "--help"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("TotalSegmentator未安装")
            return

        # 处理labeled数据
        if self.labeled_dir.exists():
            logger.info("=" * 50)
            logger.info("处理LABELED数据")
            logger.info("=" * 50)
            self.process_dataset(self.labeled_dir, "labeled_masks", "labeled")
"""""""""
        # 处理unlabeled数据
        if self.unlabeled_dir.exists():
            logger.info("=" * 50)
            logger.info("处理UNLABELED数据")
            logger.info("=" * 50)
            self.process_dataset(self.unlabeled_dir, "unlabeled_masks", "unlabeled")

        logger.info("处理完成！")
        logger.info(f"原分辨率ROI保存在: {self.output_dir}/labeled_rois 和 {self.output_dir}/unlabeled_rois")
"""""""""

if __name__ == "__main__":
    processor = PancreasROIProcessor(
        labeled_dir="/data/processed/temp_downsampled/labeled",
        #unlabeled_dir="D:/t2_pancreas_project/data/processed/temp_downsampled/unlabeled",
        output_dir="/data/processed/temp_segmentations",
        roi_margin=50
    )
    processor.run()
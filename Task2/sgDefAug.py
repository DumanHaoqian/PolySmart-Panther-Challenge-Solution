import os
import sys
import time
import psutil
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
import argparse
import warnings
import traceback
import glob

# 全局调试标志
DEBUG = True


def print_debug(message):
    """调试输出"""
    if DEBUG:
        print(f"[DEBUG] {message}")


def check_system_info():
    """检查系统信息"""
    print("\n" + "=" * 60)
    print("系统信息检查")
    print("=" * 60)

    # CPU信息
    print(f"CPU核心数: {psutil.cpu_count()}")
    print(f"CPU使用率: {psutil.cpu_percent(interval=1)}%")

    # 内存信息
    memory = psutil.virtual_memory()
    print(f"总内存: {memory.total / 1024 / 1024 / 1024:.2f} GB")
    print(f"可用内存: {memory.available / 1024 / 1024 / 1024:.2f} GB")
    print(f"内存使用率: {memory.percent}%")

    # Python版本
    print(f"Python版本: {sys.version}")

    # 包版本
    print(f"NumPy版本: {np.__version__}")
    print(f"NiBabel版本: {nib.__version__}")
    print(f"SimpleITK版本: {sitk.__version__}")
    print("=" * 60 + "\n")


def check_file_info(file_path):
    """检查文件详细信息"""
    print(f"\n检查文件: {file_path}")

    if not os.path.exists(file_path):
        print(f"错误：文件不存在！")
        return None

    # 文件大小
    file_size = os.path.getsize(file_path)
    print(f"文件大小: {file_size / 1024:.2f} KB ({file_size} bytes)")

    if file_size < 100 * 1024:  # 小于100KB
        warnings.warn("警告：文件大小异常小，可能不是完整的医学图像！")

    try:
        # 加载并检查图像
        start_time = time.time()
        img = nib.load(file_path)
        load_time = time.time() - start_time
        print(f"加载耗时: {load_time:.3f}秒")

        # 获取数据
        start_time = time.time()
        data = img.get_fdata()
        get_data_time = time.time() - start_time
        print(f"获取数据耗时: {get_data_time:.3f}秒")

        # 图像信息
        print(f"图像形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"数据范围: [{np.min(data):.2f}, {np.max(data):.2f}]")
        print(f"非零元素: {np.count_nonzero(data)} / {data.size} ({np.count_nonzero(data) / data.size * 100:.1f}%)")
        print(f"内存占用: {data.nbytes / 1024 / 1024:.2f} MB")

        # 体素间距
        spacing = img.header.get_zooms()
        print(f"体素间距: {spacing}")
        print(f"体素间距类型: {type(spacing)}")

        # 仿射矩阵
        print(f"仿射矩阵:\n{img.affine}")

        return img, data

    except Exception as e:
        print(f"错误：无法加载文件 - {str(e)}")
        traceback.print_exc()
        return None


def get_memory_usage():
    """获取当前进程内存使用"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # MB


def get_center_of_mass(mask):
    """计算mask的质心"""
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return np.array([0, 0, 0])
    return np.array([np.mean(coords[i]) for i in range(3)])


def create_smooth_deformation_field(image_shape, label_array, spacing, max_displacement=20.0):
    """创建更平滑的形变场，避免黑洞"""
    print(f"创建平滑形变场 - 最大位移: {max_displacement}mm")

    # 减小最大位移避免过度形变
    #max_displacement = min(max_displacement, 20.0)

    total_field = np.zeros((*image_shape, 3))
    unique_labels = np.unique(label_array)[1:]  # 排除背景

    if len(unique_labels) == 0:
        print("警告：没有找到前景结构，使用全局平滑形变")
        return create_global_smooth_field(image_shape, spacing, max_displacement)

    # 为每个结构创建更温和的形变
    for label_id in unique_labels:
        structure_mask = (label_array == label_id)
        if np.sum(structure_mask) == 0:
            continue

        # 更保守的随机参数
        displacement = np.random.uniform(-max_displacement * 0.5, max_displacement * 0.5, 3)
        scale_factor = np.random.uniform(0.9, 1.1, 3)  # 更温和的缩放
        rotation_angle = np.random.uniform(-5, 5)  # 小角度旋转

        # 获取结构的边界框和质心
        coords = np.array(np.where(structure_mask))
        if coords.shape[1] == 0:
            continue

        com = np.mean(coords, axis=1)

        # 计算距离权重，距离质心越远权重越小
        distances = np.sqrt(np.sum((coords - com.reshape(-1, 1)) ** 2, axis=0))
        max_dist = np.max(distances) if len(distances) > 0 else 1.0
        weights = np.exp(-distances / (max_dist * 0.3))  # 指数衰减权重

        # 应用加权形变
        rel_pos = coords.T - com
        scaled_displacement = rel_pos * (scale_factor - 1) * weights.reshape(-1, 1)
        final_displacement = scaled_displacement + displacement / spacing * weights.reshape(-1, 1)

        # 限制最大位移
        displacement_magnitude = np.linalg.norm(final_displacement, axis=1)
        max_allowed = max_displacement / np.array(spacing)
        scale_down = np.minimum(1.0, max_allowed.mean() / (displacement_magnitude + 1e-8))
        final_displacement *= scale_down.reshape(-1, 1)

        total_field[structure_mask] += final_displacement

    # 多层平滑处理
    print("应用多层平滑...")
    for i in range(3):
        # 先粗平滑
        total_field[..., i] = gaussian_filter(total_field[..., i], sigma=5.0)
        # 再细平滑
        total_field[..., i] = gaussian_filter(total_field[..., i], sigma=3.0)

    return total_field


def create_global_smooth_field(image_shape, spacing, max_displacement=10.0):
    """创建全局平滑的随机形变场"""
    # 创建低分辨率的随机场
    downsample_factor = 4
    small_shape = [s // downsample_factor for s in image_shape]

    # 生成随机位移
    small_field = np.random.uniform(-max_displacement, max_displacement, (*small_shape, 3))
    small_field /= np.array(spacing)  # 转换为体素单位

    # 上采样到原始分辨率
    full_field = np.zeros((*image_shape, 3))
    for i in range(3):
        # 使用scipy插值上采样
        from scipy.ndimage import zoom
        full_field[..., i] = zoom(small_field[..., i], downsample_factor, order=1)

    # 强力平滑
    for i in range(3):
        full_field[..., i] = gaussian_filter(full_field[..., i], sigma=5.0)

    return full_field


def apply_safe_deformation_sitk(image, displacement_field, spacing, is_label=False):
    """安全的形变应用，避免黑洞"""
    print(f"安全形变应用 - is_label: {is_label}")

    # 检查输入
    if np.any(np.isnan(displacement_field)) or np.any(np.isinf(displacement_field)):
        print("警告：位移场包含NaN或Inf值，正在修复...")
        displacement_field = np.nan_to_num(displacement_field, nan=0.0, posinf=0.0, neginf=0.0)

    # 限制位移幅度
    #max_displacement_voxels = 10.0  # 最大位移不超过10个体素
    max_displacement_voxels = 30
    displacement_magnitude = np.linalg.norm(displacement_field, axis=-1)
    mask = displacement_magnitude > max_displacement_voxels
    if np.any(mask):
        print(f"警告：检测到过大位移，正在修正 {np.sum(mask)} 个位置")
        scale_factor = max_displacement_voxels / (displacement_magnitude + 1e-8)
        for i in range(3):
            displacement_field[mask, i] *= scale_factor[mask]

    # 转换为SimpleITK格式
    sitk_image = sitk.GetImageFromArray(image.astype(np.float32))

    # 处理spacing
    if isinstance(spacing, (np.ndarray, tuple)):
        spacing_list = list(spacing)
    else:
        spacing_list = spacing
    spacing_reversed = [float(s) for s in spacing_list[::-1]]
    sitk_image.SetSpacing(spacing_reversed)

    # 创建位移场
    sitk_field = sitk.GetImageFromArray(displacement_field.astype(np.float64), isVector=True)
    sitk_field.SetSpacing(spacing_reversed)

    # 创建变换
    transform = sitk.DisplacementFieldTransform(sitk_field)

    # 选择合适的默认值和插值方法
    if is_label:
        interpolator = sitk.sitkNearestNeighbor
        default_value = 0  # 标签的背景值
    else:
        interpolator = sitk.sitkLinear
        # 使用图像的最小值作为背景值，避免黑洞
        default_value = float(np.min(image))

    print(f"使用默认值: {default_value}")

    # 应用变换
    result = sitk.Resample(
        sitk_image,
        sitk_image,
        transform,
        interpolator,
        default_value,
        sitk_image.GetPixelID()
    )

    result_array = sitk.GetArrayFromImage(result)

    # 后处理：检查并修复异常值
    if not is_label:
        # 对于图像，检查是否有异常的黑色区域
        original_min = np.min(image)
        original_max = np.max(image)

        # 如果结果中有明显低于原始最小值的区域，用周围值填充
        very_low_mask = result_array < (original_min - abs(original_min) * 0.1)
        if np.any(very_low_mask):
            print(f"检测到 {np.sum(very_low_mask)} 个异常低值，正在修复...")
            # 用中值滤波修复
            from scipy.ndimage import median_filter
            result_array[very_low_mask] = median_filter(result_array, size=3)[very_low_mask]

    return result_array


def sgDefAug_fixed(image_path, label_path, output_image_dir, output_label_dir,
                   num_augmentations=1, max_displacement=25.0):
    """修复版本的sgDefAug"""

    print(f"\n修复版sgDefAug - 最大位移: {max_displacement}mm")
    print("=" * 50)

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 加载数据
    print("加载数据...")
    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)

    image_array = image_nii.get_fdata()
    label_array = label_nii.get_fdata().astype(np.int16)
    spacing = image_nii.header.get_zooms()

    print(f"图像形状: {image_array.shape}")
    print(f"图像范围: [{np.min(image_array):.2f}, {np.max(image_array):.2f}]")
    print(f"标签唯一值: {np.unique(label_array)}")
    print(f"体素间距: {spacing}")

    base_name = os.path.basename(image_path).replace('.nii.gz', '').replace('.nii', '')

    for i in range(num_augmentations):
        print(f"\n生成增强样本 {i + 1}/{num_augmentations}")

        try:
            # 创建平滑形变场
            displacement_field = create_smooth_deformation_field(
                image_array.shape, label_array, spacing, max_displacement
            )

            print(f"位移场统计:")
            print(f"  范围: [{np.min(displacement_field):.3f}, {np.max(displacement_field):.3f}]")
            print(f"  平均幅度: {np.mean(np.linalg.norm(displacement_field, axis=-1)):.3f}")

            # 安全应用形变
            print("应用形变到图像...")
            aug_image = apply_safe_deformation_sitk(image_array, displacement_field, spacing, False)

            print("应用形变到标签...")
            aug_label = apply_safe_deformation_sitk(label_array, displacement_field, spacing, True)

            print(f"形变后图像范围: [{np.min(aug_image):.2f}, {np.max(aug_image):.2f}]")
            print(f"形变后标签唯一值: {np.unique(aug_label)}")

            # 保存结果
            aug_image_nii = nib.Nifti1Image(aug_image, image_nii.affine, image_nii.header)
            aug_label_nii = nib.Nifti1Image(aug_label.astype(np.int16), label_nii.affine, label_nii.header)

            image_output_path = os.path.join(output_image_dir, f"{base_name}_{i + 1:04d}.nii.gz")
            label_output_path = os.path.join(output_label_dir, f"{base_name}_{i + 1:04d}.nii.gz")

            nib.save(aug_image_nii, image_output_path)
            nib.save(aug_label_nii, label_output_path)

            print(f"✓ 保存完成:")
            print(f"  图像: {image_output_path}")
            print(f"  标签: {label_output_path}")

        except Exception as e:
            print(f"❌ 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n🎉 增强完成！")


def process_directory(image_dir, label_dir, output_image_dir, output_label_dir,
                      num_augmentations=10, max_displacement=40.0):
    """批量处理目录中的所有文件"""

    # 获取所有图像文件
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.nii")))

    if not image_files:
        raise ValueError(f"在 {image_dir} 中未找到 .nii 或 .nii.gz 文件")

    print(f"找到 {len(image_files)} 个图像文件")

    # 统计成功和失败的数量
    success_count = 0
    skip_count = 0

    for image_path in image_files:
        # 获取图像文件名
        image_filename = os.path.basename(image_path)

        # 从图像文件名提取ID (例如: 10303_0000.nii.gz -> 10303)
        if '_' in image_filename:
            patient_id = image_filename.split('_')[0]
        else:
            print(f"警告：无法从 {image_filename} 提取患者ID，跳过")
            skip_count += 1
            continue

        # 构建对应的标签文件路径
        possible_label_names = [
            f"{patient_id}.nii.gz",
            f"{patient_id}.nii",
            f"{patient_id}_label.nii.gz",
            f"{patient_id}_label.nii"
        ]

        label_path = None
        for label_name in possible_label_names:
            temp_path = os.path.join(label_dir, label_name)
            if os.path.exists(temp_path):
                label_path = temp_path
                break

        if not label_path:
            print(f"警告：未找到患者 {patient_id} 的标签文件，尝试过: {possible_label_names}，跳过")
            skip_count += 1
            continue

        print(f"\n处理: {image_filename} -> {os.path.basename(label_path)}")

        try:
            # 对每个文件调用原始的 sgDefAug 函数
            sgDefAug_fixed(image_path, label_path, output_image_dir, output_label_dir,
                     num_augmentations, max_displacement)
            success_count += 1
        except Exception as e:
            print(f"错误：处理 {image_filename} 时出错: {str(e)}")
            skip_count += 1
            continue

    print(f"\n处理完成！成功: {success_count}, 跳过: {skip_count}")



def process_directory_debug_1(image_dir, label_dir, output_image_dir, output_label_dir,
                            num_augmentations=10, max_displacement=40.0):
    """批量处理（带调试）"""

    print("\n" + "=" * 60)
    print("批量处理模式")
    print("=" * 60)

    # 检查系统信息
    check_system_info()


    # 获取文件列表
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.nii")))

    print(f"\n找到 {len(image_files)} 个图像文件")

    # 创建标签映射
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.nii*")))
    label_map = {}

    for label_path in label_files:
        label_filename = os.path.basename(label_path)

        # 处理不同的标签文件命名格式
        if '_0000.nii.gz' in label_filename:
            # 移除 '_0000.nii.gz' 后缀
            label_id = label_filename.replace('_0000.nii.gz', '')
        elif '_0000.nii' in label_filename:
            # 移除 '_0000.nii' 后缀
            label_id = label_filename.replace('_0000.nii', '')
        else:
            # 原有逻辑：移除 '.nii.gz' 或 '.nii'
            label_id = label_filename.replace('.nii.gz', '').replace('.nii', '')

        label_map[label_id] = label_path


    # 预览匹配
    print("\n文件匹配预览:")
    print("-" * 40)
    for i, image_path in enumerate(image_files[:5]):
        image_name = os.path.basename(image_path)
        image_id = image_name.split('_')[0] if '_' in image_name else image_name.replace('.nii.gz', '').replace('.nii',
                                                                                                                '')
        if image_id in label_map:
            print(f"✓ {image_name} -> {os.path.basename(label_map[image_id])}")
        else:
            print(f"✗ {image_name} -> 未找到匹配")

    if len(image_files) > 5:
        print(f"... 还有 {len(image_files) - 5} 个文件")

    # 确认处理
    print("\n是否开始处理? (y/n): ", end='')
    response = input().strip().lower()
    if response != 'y':
        print("已取消")
        return

    # 处理文件
    success_count = 0
    for i, image_path in enumerate(image_files):
        image_name = os.path.basename(image_path)
        image_id = image_name.split('_')[0] if '_' in image_name else image_name.replace('.nii.gz', '').replace('.nii',
                                                                                                                '')

        if image_id not in label_map:
            print(f"\n跳过 {image_name}: 未找到对应标签")
            continue

        label_path = label_map[image_id]
        print(f"\n处理 [{i + 1}/{len(image_files)}]: {image_name}")

        try:
            sgDefAug_fixed(image_path, label_path, output_image_dir, output_label_dir,
                           num_augmentations, max_displacement)
            success_count += 1
        except Exception as e:
            print(f"错误：处理失败 - {str(e)}")
            traceback.print_exc()

    print(f"\n批量处理完成: 成功 {success_count}/{len(image_files)} 个文件")


def process_directory_debug(image_dir, label_dir, output_image_dir, output_label_dir,
                            num_augmentations=10, max_displacement=40.0):
    """批量处理（带调试）"""

    print("\n" + "=" * 60)
    print("批量处理模式")
    print("=" * 60)

    # 检查系统信息
    check_system_info()

    # 获取文件列表
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.nii")))

    print(f"\n找到 {len(image_files)} 个图像文件")

    # 创建标签映射
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.nii*")))
    label_map = {}

    for label_path in label_files:
        label_filename = os.path.basename(label_path)

        # 从标签文件名提取患者ID：10xxx_0001.nii.gz -> 10xxx
        if '_' in label_filename:
            label_id = label_filename.split('_')[0]  # 取第一个下划线前的部分
        else:
            label_id = label_filename.replace('.nii.gz', '').replace('.nii', '')

        label_map[label_id] = label_path

    # 预览匹配
    print("\n文件匹配预览:")
    print("-" * 40)
    for i, image_path in enumerate(image_files[:5]):
        image_name = os.path.basename(image_path)
        # 从图像文件名提取患者ID：10xxx_0001_0000.nii.gz -> 10xxx
        if '_' in image_name:
            image_id = image_name.split('_')[0]  # 取第一个下划线前的部分
        else:
            image_id = image_name.replace('.nii.gz', '').replace('.nii', '')

        if image_id in label_map:
            print(f"✓ {image_name} -> {os.path.basename(label_map[image_id])}")
        else:
            print(f"✗ {image_name} -> 未找到匹配")

    if len(image_files) > 5:
        print(f"... 还有 {len(image_files) - 5} 个文件")

    # 确认处理
    print("\n是否开始处理? (y/n): ", end='')
    response = input().strip().lower()
    if response != 'y':
        print("已取消")
        return

    # 处理文件
    success_count = 0
    for i, image_path in enumerate(image_files):
        image_name = os.path.basename(image_path)
        # 提取图像ID
        if '_' in image_name:
            image_id = image_name.split('_')[0]
        else:
            image_id = image_name.replace('.nii.gz', '').replace('.nii', '')

        if image_id not in label_map:
            print(f"\n跳过 {image_name}: 未找到对应标签")
            continue

        label_path = label_map[image_id]
        print(f"\n处理 [{i + 1}/{len(image_files)}]: {image_name}")

        try:
            sgDefAug_fixed(image_path, label_path, output_image_dir, output_label_dir,
                           num_augmentations, max_displacement)
            success_count += 1
        except Exception as e:
            print(f"错误：处理失败 - {str(e)}")
            traceback.print_exc()

    print(f"\n批量处理完成: 成功 {success_count}/{len(image_files)} 个文件")

def main():
    parser = argparse.ArgumentParser(description='sgDefAug数据增强（调试版）')
    parser.add_argument('--mode', choices=['single', 'batch'], default='batch',
                        help='处理模式')
    parser.add_argument('--image_path', help='单文件模式：图像路径')
    parser.add_argument('--label_path', help='单文件模式：标签路径')
    parser.add_argument('--image_dir',
                        default="D:/t2_pancreas_project/data/processed/temp_segmentations/labeled_rois",
                        help='批处理模式：图像目录')
    parser.add_argument('--label_dir',
                        default="D:/t2_pancreas_project/data/processed/temp_segmentations/labeled_labels",
                        help='批处理模式：标签目录')
    parser.add_argument('--output_image_dir',
                        default="D:/t2_pancreas_project/data/processed/sgDefAug/ImagesTr",
                        help='输出图像目录')
    parser.add_argument('--output_label_dir',
                        default="D:/t2_pancreas_project/data/processed/sgDefAug/LabelsTr",
                        help='输出标签目录')
    parser.add_argument('--num_augmentations', type=int, default=1, help='增强数量')
    parser.add_argument('--max_displacement', type=float, default=60.0, help='最大位移(mm)')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')

    args = parser.parse_args()

    global DEBUG
    DEBUG = args.debug or True  # 默认开启调试

    if args.mode == 'single':
        if not args.image_path or not args.label_path:
            parser.error("单文件模式需要指定 --image_path 和 --label_path")

        check_system_info()
        sgDefAug_fixed(args.image_path, args.label_path,
                       args.output_image_dir, args.output_label_dir,
                       args.num_augmentations, args.max_displacement)
    else:
        process_directory_debug(args.image_dir, args.label_dir,
                                args.output_image_dir, args.output_label_dir,
                                args.num_augmentations, args.max_displacement)


if __name__ == "__main__":
    main()
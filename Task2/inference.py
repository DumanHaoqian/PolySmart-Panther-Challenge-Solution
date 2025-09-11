import SimpleITK as sitk
import nibabel as nib
import numpy as np
import os
import subprocess
import shutil
from pathlib import Path
from data_utils import *
from evalutils import SegmentationAlgorithm

from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import time
import glob
import warnings
warnings.filterwarnings("ignore")


class PancreaticTumorSegmentationContainer(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        # ---------- 目录/路径 ----------
        # 优先用环境变量（Dockerfile 已设置），否则回退到镜像内默认路径
        self.nnunet_model_dir = Path(os.environ.get("nnUNet_results", "/opt/algorithm/nnunet/nnUNet_results"))
        os.environ["nnUNet_results"] = str(self.nnunet_model_dir)  # 让 nnUNet 内部也一致使用同一路径
        print("[PATH] nnUNet_results =", os.environ["nnUNet_results"])

        self.work_dir = Path("/opt/algorithm/work");                 self.work_dir.mkdir(parents=True, exist_ok=True)
        self.nnunet_input_dir = Path("/opt/algorithm/nnunet/input"); self.nnunet_input_dir.mkdir(parents=True, exist_ok=True)
        self.nnunet_output_dir = Path("/opt/algorithm/nnunet/output");self.nnunet_output_dir.mkdir(parents=True, exist_ok=True)

        self.mrsegmentator_input_dir = Path("/opt/algorithm/mrsegmentator/input"); self.mrsegmentator_input_dir.mkdir(parents=True, exist_ok=True)
        self.mrsegmentator_output_dir = Path("/opt/algorithm/mrsegmentator/output"); self.mrsegmentator_output_dir.mkdir(parents=True, exist_ok=True)

        # 输入路径（只读）
        folders_with_mri = [folder for folder in os.listdir("/input/images") if "mri" in folder.lower()]
        if len(folders_with_mri) == 1:
            mr_ip_dir_name = folders_with_mri[0]
            print("Folder containing eval image:", mr_ip_dir_name)
        else:
            print("Error: Expected one folder containing 'mri', but found", len(folders_with_mri))
            mr_ip_dir_name = 'abdominal-t2-mri'  # default

        self.mr_ip_dir = Path(f"/input/images/{mr_ip_dir_name}")  # READ-ONLY

        self.output_dir = Path("/output")
        self.output_dir_images = self.output_dir / "images"
        self.output_dir_seg_mask = self.output_dir_images / "pancreatic-tumor-segmentation"
        for p in [self.output_dir, self.output_dir_images, self.output_dir_seg_mask]:
            p.mkdir(parents=True, exist_ok=True)
        self.segmentation_mask = self.output_dir_seg_mask / "tumor_seg.mha"

        # weights (平台通常把权重解压到这里)
        self.weights_path = Path("/opt/ml/model")

        # 读取第一张 .mha（只读，不写回 /input）
        mha_files = glob.glob(os.path.join(self.mr_ip_dir, '*.mha'))
        if mha_files:
            self.mr_image = mha_files[0]
        else:
            raise FileNotFoundError(f'No mha images found in input directory: {self.mr_ip_dir}')

        # nnUNet 的输入基名（输入 *_0000.nii.gz -> 输出 .nii.gz）
        self.nnunet_case_base = "cropped_mri"

        # 固定模型参数三元组（与你本地目录一致）
        self.task = "Dataset091_PantherTask2"
        self.trainer = "nnUNetTrainer"
        self.plans_name = "nnUNetResEncUNetMPlans"
        self.configuration = "3d_fullres"

    # ---------- 坐标变换辅助 ----------
    def sitk_to_nibabel_affine(self, image: sitk.Image) -> np.ndarray:
        direction = np.array(image.GetDirection(), dtype=float).reshape(3, 3)
        spacing = np.array(image.GetSpacing(), dtype=float)
        origin = np.array(image.GetOrigin(), dtype=float)
        affine = np.eye(4, dtype=float)
        affine[:3, :3] = direction @ np.diag(spacing)
        affine[:3, 3] = origin
        return affine

    def nibabel_to_sitk(self, nib_img) -> sitk.Image:
        array_xyz = np.asarray(nib_img.get_fdata(), dtype=np.float32)
        array_zyx = np.transpose(array_xyz, (2, 1, 0))
        sitk_img = sitk.GetImageFromArray(array_zyx)

        A = np.array(nib_img.affine[:3, :3], dtype=float)
        t = np.array(nib_img.affine[:3, 3], dtype=float)
        spacing = np.linalg.norm(A, axis=0); spacing[spacing == 0] = 1.0
        direction = A / spacing

        sitk_img.SetSpacing(tuple(spacing.tolist()))
        sitk_img.SetOrigin(tuple(t.tolist()))
        sitk_img.SetDirection(tuple(direction.reshape(-1).tolist()))
        return sitk_img

    def convert_to_ras(self, image: sitk.Image) -> sitk.Image:
        array = sitk.GetArrayFromImage(image)
        affine = self.sitk_to_nibabel_affine(image)
        nib_img = nib.Nifti1Image(array, affine)
        ras_img = nib.as_closest_canonical(nib_img)
        return self.nibabel_to_sitk(ras_img)

    # ---------- TS：胰腺分割（用原始图） ----------
    def run_totalsegmentator_robust(self, image: sitk.Image) -> sitk.Image:
        input_path = self.mrsegmentator_input_dir / "input.nii.gz"
        sitk.WriteImage(image, str(input_path))

        cmd = [
            "TotalSegmentator",
            "-i", str(input_path),
            "-o", str(self.mrsegmentator_output_dir),
            "--task", "total_mr"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print("TotalSegmentator STDOUT:\n", res.stdout)
        print("TotalSegmentator STDERR:\n", res.stderr)
        res.check_returncode()

        pancreas_files = list(self.mrsegmentator_output_dir.glob("*pancreas*.nii.gz"))
        if not pancreas_files:
            print("DEBUG: TS output dir listing:", list(self.mrsegmentator_output_dir.rglob("*")))
            raise FileNotFoundError("No pancreas mask found from TotalSegmentator")

        pancreas_seg = sitk.ReadImage(str(pancreas_files[0]))
        self._log_mask_stats(pancreas_seg, title="pancreas_raw")

        pan_arr = sitk.GetArrayFromImage(pancreas_seg)
        pan_bin = (pan_arr > 0).astype(np.uint8)
        pancreas_mask = sitk.GetImageFromArray(pan_bin); pancreas_mask.CopyInformation(pancreas_seg)
        self._log_mask_stats(pancreas_mask, title="pancreas_binary")

        # 不删除 TS 输出，便于调试（稳定后可改为清理）
        try:
            input_path.unlink(missing_ok=True)
        except TypeError:
            if input_path.exists():
                input_path.unlink()

        return pancreas_mask

    # ---------- 主流程 ----------
    def run(self):
        _show_torch_cuda_info()
        t0 = time.perf_counter()

        # 1) 准备 nnUNet 模型目录 & checkpoint
        self.move_checkpoints(self.weights_path, folds="0,1,2,3,4")
        self._ensure_model_ready(self.task, self.trainer, self.plans_name, self.configuration)

        # 2) 读取原始影像
        original_image = sitk.ReadImage(self.mr_image)

        # 3) TS 分割胰腺（用原图）
        pancreas_mask = self.run_totalsegmentator_robust(original_image)

        # 4) 空 mask 回退（避免崩）
        if np.count_nonzero(sitk.GetArrayFromImage(pancreas_mask)) == 0:
            print("WARNING: Empty pancreas mask from TS. Using whole image (no ROI crop).")
            cropped_mri = original_image
            crop_coordinates = ((0, 0, 0), original_image.GetSize())
        else:
            margins = [50, 50, 50]
            cropped_mri, crop_coordinates = CropPancreasROI(original_image, pancreas_mask, margins)

        # 5) 保存裁剪图给 nnUNet（*_0000.nii.gz）
        nnunet_in_path = self.nnunet_input_dir / f"{self.nnunet_case_base}_0000.nii.gz"
        sitk.WriteImage(cropped_mri, str(nnunet_in_path))
        print("NNUnet input dir listing:", os.listdir(self.nnunet_input_dir))
        assert nnunet_in_path.exists(), f"{nnunet_in_path.name} not found in {self.nnunet_input_dir}"

        # 6) nnUNet 预测
        print(f"Input dir:{self.nnunet_input_dir}, output dir:{self.nnunet_output_dir}, task:{self.task}, folds: 0")
        self.predict(
            input_dir=self.nnunet_input_dir,
            output_dir=self.nnunet_output_dir,
            task=self.task
        )
        print(f"Output files: {os.listdir(self.nnunet_output_dir)}")

        # 7) 读取输出并还原到原图尺寸
        mr_mask_name = f"{self.nnunet_case_base}.nii.gz"
        nnunet_out_path = self.nnunet_output_dir / mr_mask_name
        if not nnunet_out_path.exists():
            raise FileNotFoundError(
                f"nnUNet output not found: {nnunet_out_path}. "
                f"Check case naming '*_0000.nii.gz' and nnUNet logs."
            )

        tumor_mask_cropped = sitk.ReadImage(str(nnunet_out_path))
        print(f"nnUNet output shape: {tumor_mask_cropped.GetSize()}, spacing: {tumor_mask_cropped.GetSpacing()}")
        tumor_array = sitk.GetArrayFromImage(tumor_mask_cropped)

        final_tumor_mask = restore_to_full_size(
            sitk.GetImageFromArray(tumor_array),
            original_image,
            crop_coordinates
        )

        # 8) 输出
        final_tumor_array = sitk.GetArrayFromImage(final_tumor_mask).astype(np.uint8)
        final_tumor_mask_u8 = sitk.GetImageFromArray(final_tumor_array)
        final_tumor_mask_u8.CopyInformation(original_image)
        sitk.WriteImage(final_tumor_mask_u8, str(self.segmentation_mask))

        print(f"Processing time: {time.perf_counter() - t0:.3f}s")

    # ---------- nnUNet 预测 ----------
    def predict(self, input_dir, output_dir, task="Dataset091_PantherTask2", trainer="nnUNetTrainer",
                configuration="3d_fullres", checkpoint="checkpoint_final.pth", folds="0,1,2,3,4",
                plans_file="nnUNetResEncUNetMPlans"):
        # 使用与 __init__ 一致的路径
        os.environ['nnUNet_results'] = str(self.nnunet_model_dir)

        # 预测前做一次存在性检查
        self._ensure_model_ready(task, trainer, plans_file, configuration)

        cmd = [
            'nnUNetv2_predict',
            '-d', task,
            '-i', str(input_dir),
            '-o', str(output_dir),
            '-c', configuration,
            '-tr', trainer,
            '--disable_progress_bar',
            '--continue_prediction'
        ]

        if folds:
            cmd.append('-f')
            fold_list = folds.split(',') if isinstance(folds, str) and ',' in folds else [folds]
            cmd.extend(fold_list)

        if checkpoint:
            cmd.append('-chk'); cmd.append(str(checkpoint))

        if plans_file is not None:
            cmd.append('-p'); cmd.append(str(plans_file))

        cmd_str = " ".join(cmd)
        print(f"Running command: {cmd_str}")
        subprocess.check_call(cmd_str, shell=True)

    # ---------- 拷贝 checkpoint 到正确目录 ----------
    def move_checkpoints(self, source_dir, folds="0,1,2,3,4", trainer="nnUNetTrainer", task="Dataset091_PantherTask2"):
        os.makedirs(self.nnunet_model_dir, exist_ok=True)
        print("Weights dir listing:", os.listdir(source_dir) if Path(source_dir).exists() else "MISSING")
        task_name = task.split("_")[1]
        fold_list = folds.split(',') if isinstance(folds, str) and ',' in folds else [str(folds)]

        for fold in fold_list:
            src = os.path.join(source_dir, f"checkpoint_best_{task_name}_fold_{fold}.pth")
            dst = os.path.join(
                self.nnunet_model_dir, task, f"{trainer}__{self.plans_name}__{self.configuration}",
                f"fold_{fold}", "checkpoint_final.pth"
            )
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.copyfile(src, dst)
                print(f"Copied checkpoint for fold {fold} to {dst}")
            except FileNotFoundError:
                print(f"[WARN] Source checkpoint not found: {src}")
            except Exception as e:
                print(f"[WARN] Error moving checkpoint for fold {fold}: {e}")

    # ---------- 确保模型元数据就绪 ----------
    def _ensure_model_ready(self, task: str, trainer: str, plans: str, configuration: str):
        root = Path(self.nnunet_model_dir)
        model_dir = root / task / f"{trainer}__{plans}__{configuration}"
        parent = model_dir.parent
        print("Model dir resolved to:", model_dir)

        # dataset.json: 有时在 model_dir，有时在父级 task 目录；确保最终在 model_dir 也有一份
        ds_in_model = model_dir / "dataset.json"
        ds_in_parent = parent / "dataset.json"

        if not ds_in_model.exists():
            if ds_in_parent.exists():
                shutil.copyfile(ds_in_parent, ds_in_model)
                print(f"Copied dataset.json -> {ds_in_model}")
            else:
                raise FileNotFoundError(f"dataset.json not found in {ds_in_model} or {ds_in_parent}")

        # plans：v2 可能是 plans.json 或 plans.pkl，任意其一存在即可
        plans_json = model_dir / "plans.json"
        plans_pkl = model_dir / "plans.pkl"
        if not (plans_json.exists() or plans_pkl.exists()):
            # 兼容：若父级有，也拷过来
            pj_parent = parent / "plans.json"
            pk_parent = parent / "plans.pkl"
            if pj_parent.exists():
                shutil.copyfile(pj_parent, plans_json)
                print(f"Copied plans.json -> {plans_json}")
            elif pk_parent.exists():
                shutil.copyfile(pk_parent, plans_pkl)
                print(f"Copied plans.pkl -> {plans_pkl}")
            else:
                raise FileNotFoundError(f"Neither plans.json nor plans.pkl found under {model_dir} (or its parent).")

    # ---------- 日志 ----------
    def _log_mask_stats(self, mask_img: sitk.Image, title: str = "pancreas_mask"):
        arr = sitk.GetArrayFromImage(mask_img)
        uniq, cnt = np.unique(arr, return_counts=True)
        sp = np.array(mask_img.GetSpacing(), dtype=float)
        voxel_vol_mm3 = sp[0] * sp[1] * sp[2]
        voxel_vol_ml = voxel_vol_mm3 / 1000.0

        print(f"[MaskStats::{title}] spacing(x,y,z) = {tuple(sp.tolist())}, voxel_volume = {voxel_vol_mm3:.3f} mm^3 ({voxel_vol_ml:.6f} ml)")
        print(f"[MaskStats::{title}] unique labels and voxel counts:")
        for u, c in zip(uniq.tolist(), cnt.tolist()):
            print(f"  - label={int(u):d}, voxels={c}, volume={c * voxel_vol_ml:.3f} ml")

        bin_img = sitk.BinaryThreshold(mask_img, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0)
        cc = sitk.ConnectedComponent(bin_img)
        stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(cc)
        num_cc = len(stats.GetLabels())
        print(f"[MaskStats::{title}] connected components (foreground>0): {num_cc}")
        if num_cc > 0:
            comp = []
            for lab in stats.GetLabels():
                vox = int(stats.GetNumberOfPixels(lab))
                comp.append((lab, vox, vox * voxel_vol_ml))
            comp.sort(key=lambda x: x[1], reverse=True)
            for lab, vox, vol_ml in comp[:10]:
                print(f"  - CC label={lab}, voxels={vox}, volume={vol_ml:.3f} ml")


def _show_torch_cuda_info():
    import torch
    print("=+=" * 10)
    print(torch.__version__)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    PancreaticTumorSegmentationContainer().run()

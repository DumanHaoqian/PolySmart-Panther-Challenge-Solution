import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
import torch.nn.functional as F
####################### New Added Boundary Loss #########################################################
class SignedBoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

        # 3D Laplacian kernel for edge detection
        self.kernel_3d = torch.tensor([
            [[[0, 0, 0],
              [0, -1, 0],
              [0, 0, 0]],

             [[0, -1, 0],
              [-1, 6, -1],
              [0, -1, 0]],

             [[0, 0, 0],
              [0, -1, 0],
              [0, 0, 0]]]
        ], dtype=torch.float32).unsqueeze(0)  # shape: [1, 1, 3, 3, 3]

    def forward(self, pred_soft: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred_soft: [B, C, D, H, W]  (softmaxed network output)
        target:    [B, 1, D, H, W]  (class indices)
        """
        device = pred_soft.device
        kernel = self.kernel_3d.to(device)

        # Extract foreground mask
        fg_mask = (target == 1).float()  # [B, 1, D, H, W]

        # Detect boundary: edge = where Laplacian != 0
        edge = F.conv3d(fg_mask, kernel, padding=1).abs() > 0  # [B, 1, D, H, W]

        # Create signed map: +1 outside, +1.5 on boundary, -1 inside
        signed_boundary = torch.where(
            edge, torch.tensor(1.5, device=device),
            torch.where(fg_mask.bool(), torch.tensor(-1.0, device=device), torch.tensor(1.0, device=device))
        )  # [B, 1, D, H, W]

        # Predict foreground prob
        fg_pred = pred_soft[:, 1:2]  # [B, 1, D, H, W]

        # Compute loss: average of signed difference
        loss = torch.mean(fg_pred * signed_boundary)

        return loss

class DC_CE_Boundary_loss(nn.Module):
    def __init__(self,
                 soft_dice_kwargs: dict,
                 ce_kwargs: dict,
                 alpha: float = 0.5,
                 beta: float = 0.3,
                 dice_class=MemoryEfficientSoftDiceLoss,
                 class_weights: list = None):
        """
        Composite loss: α·Dice + β·CE + (1−α−β)·Boundary
        :param soft_dice_kwargs: dict, args for Dice
        :param ce_kwargs: dict, args for CE (e.g., ignore_index)
        :param class_weights: list of floats, e.g., [w_bg, w_fg]
        """
        super().__init__()
        assert 0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0 and alpha + beta <= 1.0
        self.alpha = alpha
        self.beta = beta

        if class_weights is not None:
            ce_kwargs = ce_kwargs.copy()  # avoid modifying external dict
            ce_kwargs['weight'] = torch.tensor(class_weights, dtype=torch.float32)

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.boundary = SignedBoundaryLoss()

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if hasattr(self.ce, 'weight') and self.ce.weight is not None:
            self.ce.weight = self.ce.weight.to(net_output.device)    
        
        dice_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target[:, 0])
        pred_soft = softmax_helper_dim1(net_output)
        b_loss = self.boundary(pred_soft, target)
        w_b = 1.0 - self.alpha - self.beta
        #return self.alpha * dice_loss + self.beta * ce_loss + w_b * b_loss
        #return self.alpha * dice_loss + (1-self.alpha) * ce_loss 
        return  dice_loss  
        
#######################################################################################################
class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

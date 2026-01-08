# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Union

import torch
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector


@MODELS.register_module()
class YOLOX(SingleStageDetector):
    r"""Implementation of `YOLOX: Exceeding YOLO Series in 2021
    <https://arxiv.org/abs/2107.08430>`_

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLOX. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLOX. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

    def extract_cam_features(self,
                            x: torch.Tensor,
                            target_layer: str = 'neck') -> Union[torch.Tensor, List[torch.Tensor]]:
        """Extract features for CAM (Class Activation Mapping) computation.

        This method extracts intermediate features from specified layers
        for visualization and interpretation purposes.

        Args:
            x (torch.Tensor): Input image tensor of shape (N, C, H, W)
            target_layer (str): Target layer for feature extraction.
                Options: 'backbone', 'neck', 'head'
                - 'backbone': Returns backbone output features
                - 'neck': Returns neck output (multi-scale features)
                - 'head': Not recommended, would require head modifications
                Defaults to 'neck'

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]:
                - If target_layer='backbone': Returns tensor of shape (N, C, H, W)
                - If target_layer='neck': Returns list of tensors for multi-scale features

        Examples:
            >>> model = init_model('yolox_s.py', 'yolox_s.pth')
            >>> img = torch.rand(1, 3, 640, 640)
            >>> # Get neck features (multi-scale)
            >>> neck_feats = model.extract_cam_features(img, 'neck')
            >>> print(len(neck_feats))  # 3 (for 3 scales)
            >>> # Get backbone features
            >>> backbone_feats = model.extract_cam_features(img, 'backbone')
        """
        if target_layer == 'backbone':
            # Extract backbone features
            x = self.backbone(x)
            return x
        elif target_layer == 'neck':
            # Extract neck features (multi-scale)
            x = self.extract_feat(x)
            return x
        elif target_layer == 'head':
            # Head extraction is more complex as it involves multiple branches
            # For now, we return neck features as head's input
            warnings.warn("Extracting head features is not fully supported. "
                         "Returning neck features instead.")
            x = self.extract_feat(x)
            return x
        else:
            raise ValueError(f"Unsupported target_layer: {target_layer}. "
                           f"Choose from ['backbone', 'neck'].")

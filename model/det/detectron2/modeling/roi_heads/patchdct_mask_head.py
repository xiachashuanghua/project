import fvcore.nn.weight_init as weight_init
from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from .mask_head import ROI_MASK_HEAD_REGISTRY
from detectron2.config import configurable
from .mask_head import BaseMaskRCNNHead
from detectron2.layers import Conv2d, ShapeSpec, cat, get_norm, ConvTranspose2d
from detectron2.structures import Instances

from .mask_encoding import DctMaskEncoding
from .utils import patch2masks, masks2patch
from .get_gt_info import GT_infomation


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNPatchDCTHead(BaseMaskRCNNHead):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="",
                 dct_vector_dim, mask_size, pooler_resolution, hidden_features, fine_features_resolution,
                 mask_size_assemble, patch_size, patch_dct_vector_dim, mask_loss_para, dct_loss_type,
                 num_stage, mask_loss_para_each_stage, patch_threshold, eval_gt,
                 **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_classes (int): the number of classes. 1 if using class agnostic prediction.For COCO,num_classes=80
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
            dct_vector_dim: dct vector dim in DCT_MASK(default=300)
            mask_size: resolution of mask to be refined(default=112)
            hidden_features: feature dim of linear layer(default=1024)
            fine_features_resolution: feature map in PatchDCT(default=42)
            mask_size_assemble: mask size in PatchDCT(default=112)
            patch_size: patch size(default=8)
            patch_dct_vector_dim: DCT vector dim for each patch
            mask_loss_para: coefficient of total loss(default=1)
            dct_loss_type: loss type of DCT vector regressor(default=l1, option=[l1, sl1,l2])
            num_stage: number of segmentation stage, equals to number of PatchDCT blocks+1(default=2)
            mask_loss_para_each_stage: coefficient of loss for each segmentation stage
            patch_threshold: threshold used for classifier(default=0.3),
            eval_gt: use for calculate the upper bound of the model

        """
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.patch_dct_vector_dim = patch_dct_vector_dim
        self.mask_size_assemble = mask_size_assemble
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.dct_vector_dim = dct_vector_dim
        self.mask_size = mask_size
        self.dct_loss_type = dct_loss_type
        self.mask_loss_para = mask_loss_para
        self.scale = self.mask_size // self.patch_size
        self.ratio = fine_features_resolution // self.scale
        self.patch_threshold = patch_threshold
        self.eval_gt = eval_gt
        self.num_stage = num_stage - 1
        self.loss_para = mask_loss_para_each_stage
        print("num stage of the model is {}".format(self.num_stage))

        self.dct_encoding = DctMaskEncoding(vec_dim=self.dct_vector_dim, mask_size=self.mask_size)
        self.patch_dct_encoding = DctMaskEncoding(vec_dim=self.patch_dct_vector_dim, mask_size=self.patch_size)
        self.gt = GT_infomation(self.mask_size_assemble, self.mask_size, self.patch_size, self.scale,
                                self.dct_encoding, self.patch_dct_encoding)

        self.conv_norm_relus = []

        cur_channels = input_shape.channels
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        self.predictor = nn.Sequential(
            nn.Linear(14 ** 2 * conv_dim, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.dct_vector_dim)
        )
        self.reshape = Conv2d(
            1,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=F.relu
        )
        self.fusion = nn.Sequential(
            Conv2d(cur_channels,
                   conv_dim,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   bias=not conv_norm,
                   norm=get_norm(conv_norm, conv_dim),
                   activation=F.relu),
            Conv2d(cur_channels,
                   conv_dim,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   bias=not conv_norm,
                   norm=get_norm(conv_norm, conv_dim),
                   activation=F.relu)
        )

        self.downsample = nn.Sequential(
            Conv2d(
                cur_channels,
                self.hidden_features,
                kernel_size=self.ratio,
                stride=self.ratio,
                padding=0,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu, ),
            Conv2d(self.hidden_features,
                   self.hidden_features,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   bias=not conv_norm,
                   norm=get_norm(conv_norm, conv_dim),
                   activation=F.relu),
        )

        self.predictor1 = Conv2d(self.hidden_features,
                                 self.patch_dct_vector_dim * self.num_classes,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 )
        self.predictor_bfg = Conv2d(self.hidden_features,
                                    3 * self.num_classes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    )

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
            hidden_features=cfg.MODEL.ROI_MASK_HEAD.HIDDEN_FEATURES,
            pooler_resolution=cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION,
            dct_vector_dim=cfg.MODEL.ROI_MASK_HEAD.DCT_VECTOR_DIM,
            mask_loss_para=cfg.MODEL.ROI_MASK_HEAD.MASK_LOSS_PARA,
            mask_size=cfg.MODEL.ROI_MASK_HEAD.MASK_SIZE,
            dct_loss_type=cfg.MODEL.ROI_MASK_HEAD.DCT_LOSS_TYPE,
            fine_features_resolution=cfg.MODEL.ROI_MASK_HEAD.FINE_FEATURES_RESOLUTION,
            mask_size_assemble=cfg.MODEL.ROI_MASK_HEAD.MASK_SIZE_ASSEMBLE,
            patch_size=cfg.MODEL.ROI_MASK_HEAD.PATCH_SIZE,
            patch_dct_vector_dim=cfg.MODEL.ROI_MASK_HEAD.PATCH_DCT_VECTOR_DIM,
            num_stage=cfg.MODEL.ROI_MASK_HEAD.NUM_STAGE,
            mask_loss_para_each_stage=cfg.MODEL.ROI_MASK_HEAD.MASK_LOSS_PARA_EACH_STAGE,
            patch_threshold=cfg.MODEL.ROI_MASK_HEAD.PATCH_THRESHOLD,
            eval_gt=cfg.MODEL.ROI_MASK_HEAD.EVAL_GT,
        )

        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return ret

    def layers(self, x, fine_mask_features, instances):
        """

        Args:
            x: feature map used in DCT-Mask
            fine_mask_features: feature map used in PatchDCT

        Returns:
            x (Tensor): [B, D]. D is dct-dim. [B, D]. DCT_Vector in DCT-Mask.
            bfg: A tensor of shape [B,num_class,3,scale,scale] or a dict of tensors(for multi-stage PatchDCT)
                A NxN masks is divided into scale x scale patches.
                bfg demonstrates results of three-class classifier in PatchDCT
                0 for foreground,1 for mixed,2 for background
            patch_vectors : A tensor of shape:[B,num_class,patch_dct_vector_dim,scale,scale] or a dict of tensors(for multi-stage PatchDCT)
                    DCT vector for each patch (only calculate loss for mixed patch)
        """
        for layer in self.conv_norm_relus:
            x = layer(x)
        # DCT-Mask 
        x = self.predictor(x.flatten(start_dim=1))
        if not self.training:
            num_masks = x.shape[0]
            if num_masks == 0:
                return x, 0, 0
        # reverse transform to obtain high-resolution masks
        masks = self.dct_encoding.decode(x).real.reshape(-1, 1, self.mask_size, self.mask_size)

        # PatchDCT
        bfg, patch_vectors = self.patchdct(masks, fine_mask_features)

        if self.num_stage == 1:
            return x, bfg, patch_vectors

        else:
            # for multi-stage PatchDCT
            if self.training:
                classes = self.gt.get_gt_classes(instances)
            else:
                classes = instances[0].pred_classes
            num_instance = classes.size()[0]
            indices = torch.arange(num_instance)
            bfg = bfg[indices, classes].permute(0, 2, 3, 1).reshape(-1, 3)
            patch_vectors = patch_vectors[indices, classes].permute(0, 2, 3, 1).reshape(-1, self.patch_dct_vector_dim)

            bfg_dict = {}
            patch_vectors_dict = {}
            bfg_dict[0] = bfg
            patch_vectors_dict[0] = patch_vectors
            for i in range(1, self.num_stage):
                masks = self.stage_patch2mask(bfg, patch_vectors)
                bfg, patch_vectors = self.patchdct(masks, fine_mask_features)
                bfg = bfg[indices, classes].permute(0, 2, 3, 1).reshape(-1, 3)
                patch_vectors = patch_vectors[indices, classes].permute(0, 2, 3, 1).reshape(-1,
                                                                                            self.patch_dct_vector_dim)
                bfg_dict[i] = bfg
                patch_vectors_dict[i] = patch_vectors
            return x, bfg_dict, patch_vectors_dict

    def stage_patch2mask(self, bfg, patch_vectors):
        device = bfg.device
        index = torch.argmax(bfg, dim=1)
        bg = torch.zeros_like(patch_vectors, device=device)
        bg[index == 1] = 1
        fg = torch.zeros_like(patch_vectors, device=device)
        fg[index == 2, 0] = self.patch_size
        masks = patch_vectors * bg + fg
        masks = self.patch_dct_encoding.decode(masks).real
        masks = patch2masks(masks, self.scale, self.patch_size, self.mask_size_assemble)
        return masks[:, None, :, :]

    def patchdct(self, masks, fine_mask_features):
        """
        PatchDCT block
        Args:
            fine_mask_features: feature map cropped from FPN P2
            masks: masks to be refined

        Returns:
            bfg and patch_vector of each PatchDCT block
        """
        masks = F.interpolate(masks, size=(self.scale * self.ratio, self.scale * self.ratio))
        masks = self.reshape(masks)
        fine_mask_features = masks + fine_mask_features
        fine_mask_features = self.fusion(fine_mask_features)
        fine_mask_features = self.downsample(fine_mask_features)
        patch_vectors = self.predictor1(fine_mask_features)
        bfg = self.predictor_bfg(fine_mask_features)
        bfg = bfg.reshape(-1, self.num_classes, 3, self.scale, self.scale)
        patch_vectors = patch_vectors.reshape(-1, self.num_classes, self.patch_dct_vector_dim, self.scale, self.scale)
        return bfg, patch_vectors

    def forward(self, x, fine_mask_features, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x, bfg, patch_vectors = self.layers(x, fine_mask_features, instances)
        if self.training:
            return {"loss_mask": self.mask_rcnn_dct_loss(x, bfg, patch_vectors, instances, self.vis_period)}
        else:
            pred_instances = self.mask_rcnn_dct_inference(x, bfg, patch_vectors, instances)
            return pred_instances

    def mask_rcnn_dct_loss(self, pred_mask_logits, bfg, patch_vectors, instances, vis_period=0):
        """
        Compute the mask prediction loss defined in the Mask R-CNN paper.

        Args:
            pred_mask_logits (Tensor): [B, D]. D is dct-dim. [B, D]. DCT_Vector in DCT-Mask.
            bfg: A tensor of shape [B,num_class,3,scale,scale] or a dict of tensors(for multi-stage PatchDCT)
                A NxN masks is divided into scale x scale patches.
                bfg demonstrates results of three-class classifier in PatchDCT
                0 for foreground,1 for mixed,2 for background
            patch_vectors : A tensor of shape:[B,num_class,patch_dct_vector_dim,scale,scale] or a dict of tensors(for multi-stage PatchDCT)
                    DCT vector for each patch (only calculate loss for mixed patch)
            instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. These instances are in 1:1
                correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
                ...) associated with each instance are stored in fields.
            vis_period (int): the period (in steps) to dump visualization.

        Returns:
            mask_loss (Tensor): A scalar tensor containing the loss.
        """

        if self.dct_loss_type == "l1":
            loss_func = F.l1_loss
        elif self.dct_loss_type == "sl1":
            loss_func = F.smooth_l1_loss
        elif self.dct_loss_type == "l2":
            loss_func = F.mse_loss
        else:
            raise ValueError("Loss Type Only Support : l1, l2; yours: {}".format(self.dct_loss_type))

        gt_masks, gt_classes, gt_masks_coarse, gt_bfg = self.gt.get_gt_mask(instances, pred_mask_logits)

        mask_loss = self.loss_para[0] * loss_func(pred_mask_logits, gt_masks_coarse)

        if self.num_stage == 1:

            num_instance = gt_classes.size()[0]
            indice = torch.arange(num_instance)
            bfg = bfg[indice, gt_classes].permute(0, 2, 3, 1).reshape(-1, 3)
            patch_vectors = patch_vectors[indice, gt_classes].permute(0, 2, 3, 1).reshape(-1, self.patch_dct_vector_dim)
            patch_vectors = patch_vectors[gt_bfg == 1, :]
            mask_loss_2 = F.cross_entropy(bfg, gt_bfg)
            mask_loss_3 = loss_func(patch_vectors, gt_masks)
            mask_loss = mask_loss + self.loss_para[1] * (mask_loss_2 + mask_loss_3)
            mask_loss = self.mask_loss_para * mask_loss
        else:
            for i in range(self.num_stage):
                bfg_this_stage = bfg[i]
                patch_vectors_this_stage = patch_vectors[i]
                patch_vectors_this_stage = patch_vectors_this_stage[gt_bfg == 1]
                mask_loss += self.loss_para[i + 1] * (
                            F.cross_entropy(bfg_this_stage, gt_bfg) + loss_func(patch_vectors_this_stage, gt_masks))
        return mask_loss

    def mask_rcnn_dct_inference(self, pred_mask_logits, bfg, patch_vectors, pred_instances):
        """
        Convert pred_mask_logits to estimated foreground probability masks while also
        extracting only the masks for the predicted classes in pred_instances. For each
        predicted box, the mask of the same class is attached to the instance by adding a
        new "pred_masks" field to pred_instances.

        Args:
            pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
                for class-specific or class-agnostic, where B is the total number of predicted masks
                in all images, C is the number of foreground classes, and Hmask, Wmask are the height
                and width of the mask predictions. The values are logits.
            bfg: A tensor of shape [B,num_class,3,scale,scale] or a dict of tensors(for multi-stage PatchDCT)
                A NxN masks is divided into scale x scale patches.
                bfg demonstrates results of three-class classifier in PatchDCT
                0 for foreground,1 for mixed,2 for background
            patch_vectors : A tensor of shape:[B,num_class,patch_dct_vector_dim,scale,scale] or a dict of tensors(for multi-stage PatchDCT)
                    DCT vector for each patch (only calculate loss for mixed patch)
            pred_instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. Each Instances must have field "pred_classes".

        Returns:
            None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
                Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
                masks the resolution predicted by the network; post-processing steps, such as resizing
                the predicted masks to the original image resolution and/or binarizing them, is left
                to the caller.
        """

        num_patch = pred_mask_logits.shape[0]
        device = pred_mask_logits.device
        if num_patch == 0:
            pred_instances[0].pred_masks = torch.empty([0, 1, self.mask_size, self.mask_size]).to(device)
            return pred_instances
        else:

            pred_classes = pred_instances[0].pred_classes
            num_masks = pred_classes.shape[0]
            indices = torch.arange(num_masks)
            if self.num_stage > 1:
                bfg = bfg[self.num_stage - 1]
                patch_vectors = patch_vectors[self.num_stage - 1]
            else:
                bfg = bfg[indices, pred_classes].permute(0, 2, 3, 1).reshape(-1, 3)
                patch_vectors = patch_vectors[indices, pred_classes].permute(0, 2, 3, 1).reshape(-1,
                                                                                                 self.patch_dct_vector_dim)

            with torch.no_grad():

                bfg = F.softmax(bfg, dim=1)
                bfg[bfg[:, 0] > self.patch_threshold, 0] = bfg[bfg[:, 0] > self.patch_threshold, 0] + 1
                bfg[bfg[:, 2] > self.patch_threshold, 2] = bfg[bfg[:, 2] > self.patch_threshold, 2] + 1
                index = torch.argmax(bfg, dim=1)

                if self.eval_gt:
                    gt_masks, index = self.gt.get_gt_mask_inference(pred_instances, pred_mask_logits)
                    patch_vectors[index == 1] = gt_masks

                patch_vectors[index == 0, ::] = 0
                patch_vectors[index == 2, ::] = 0
                patch_vectors[index == 2, 0] = self.patch_size

                pred_mask_rc = self.patch_dct_encoding.decode(patch_vectors)
                # assemble patches to obtain an entire mask
                pred_mask_rc = patch2masks(pred_mask_rc, self.scale, self.patch_size, self.mask_size_assemble)

            pred_mask_rc = pred_mask_rc[:, None, :, :]
            pred_instances[0].pred_masks = pred_mask_rc
            return pred_instances

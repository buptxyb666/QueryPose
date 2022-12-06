# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import List

import numpy as np
import torch 
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.modeling.roi_heads import build_roi_heads

from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss

from .loss import SetCriterion, HungarianMatcher
from .head import Decoder
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, gaussian_radius, draw_umich_gaussian
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

__all__ = ["SparseRCNN"]

def make_sine_position_embedding(feature, temperature=10000,
                                  scale=2 * math.pi):
    h, w = feature.shape[2], feature.shape[3]
    d_model = feature.shape[1]
    area = torch.ones(1, h, w).to(feature.device)  # [b, h, w]
    y_embed = area.cumsum(1, dtype=torch.float32)
    x_embed = area.cumsum(2, dtype=torch.float32)

    one_direction_feats = d_model // 2

    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(one_direction_feats, dtype=torch.float32).to(feature.device)
    dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack(
        (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    return pos



@META_ARCH_REGISTRY.register()
class QueryPose(nn.Module):
    """
    Implement SparseRCNN
    """

    def __init__(self, cfg):
        super().__init__()
        # import pudb;pudb.set_trace()
        self.device = torch.device(cfg.MODEL.DEVICE) 

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        self.num_proposals = cfg.MODEL.SparseRCNN.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        self.num_heads = cfg.MODEL.SparseRCNN.NUM_HEADS
        self.num_parts = cfg.MODEL.QueryPose.NUM_PART
        self.part_dim = cfg.MODEL.QueryPose.PART_DIM 
        self.num_kps = cfg.MODEL.QueryPose.NUM_KPS
        
        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        
        # Build Proposals.
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.part_query = nn.Embedding(self.num_parts, self.part_dim)
        
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)
        # nn.init.constant_(self.part_embed.weight, 0.0)
        # Build decoder.
        self.head = Decoder(cfg=cfg, roi_input_shape=self.backbone.output_shape())

        # Loss parameters:
        class_weight = cfg.MODEL.SparseRCNN.CLASS_WEIGHT
        giou_weight = cfg.MODEL.SparseRCNN.GIOU_WEIGHT
        l1_weight = cfg.MODEL.SparseRCNN.L1_WEIGHT
        kps_weight = 1.0
        no_object_weight = cfg.MODEL.SparseRCNN.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.SparseRCNN.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        self.aux_hm = cfg.MODEL.QueryPose.AUX_KPS_HEATMAP
        self.aux_hm_weight = cfg.MODEL.QueryPose.AUX_KPS_HEATMAP_WEIGHT


        if self.aux_hm:
            self.hm_fc = nn.Sequential(
                        nn.Conv2d(self.hidden_dim, self.hidden_dim,
                        kernel_size=3, padding=1, bias=True), 
                        nn.ReLU(inplace=True),
                        # nn.Conv2d(self.hidden_dim, self.hidden_dim,
                        # kernel_size=3, padding=1, bias=True),
                        # nn.ReLU(inplace=True),
                        # nn.Conv2d(self.hidden_dim, self.hidden_dim,
                        # kernel_size=3, padding=1, bias=True),
                        # nn.ReLU(inplace=True),
                        nn.Conv2d(self.hidden_dim, self.num_kps, 
                        kernel_size=1, stride=1, 
                        padding=0, bias=True))
            
            nn.init.constant_(self.hm_fc[-1].bias, -2.19)
        

        # Build Criterion.
        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=class_weight, 
                                   cost_bbox=l1_weight, 
                                   cost_giou=giou_weight,
                                   use_focal=self.use_focal)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight, "loss_kps": kps_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes", "keypoints"]

        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=no_object_weight,
                                      losses=losses,
                                      use_focal=self.use_focal)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)


    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()        
        for f in self.in_features:
            feature = src[f]
            features.append(feature)
        

        # Prepare Proposals.
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]

        # Prediction.
        outputs_class, outputs_coord, outputs_kps, outputs_sgm = self.head(features, proposal_boxes, self.init_proposal_features.weight, self.part_query.weight)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_keypoints':outputs_kps[-1], 'pred_sgm':outputs_sgm[-1]}

        if self.training:
            #aux_heatmap
            if self.aux_hm:
                heatmap = self.hm_fc(features[0])
            else:
                heatmap = None

            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, features[0].shape)
            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b, 'pred_keypoints': c, 'pred_sgm': d}
                                         for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_kps[:-1], outputs_sgm[:-1])]
            
            loss_dict = self.criterion(output, heatmap, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
                if k == 'loss_aux_hm':
                    loss_dict[k] *= self.aux_hm_weight
            return loss_dict

        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            kps_pred = output['pred_keypoints']
            
            results = self.inference(box_cls, box_pred, kps_pred, images.image_sizes)
            
            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])

                    offset_x = results_per_image.pred_boxes.tensor[:, 0:1]
                    offset_y = results_per_image.pred_boxes.tensor[:, 1:2]
                    roi_w = results_per_image.pred_boxes.tensor[:,2:3] - offset_x
                    roi_h = results_per_image.pred_boxes.tensor[:,3:4] - offset_y
                    results_per_image.pred_keypoints[:, :, 0] = (results_per_image.pred_keypoints[:, :, 0] + 0.5) * roi_w + offset_x
                    results_per_image.pred_keypoints[:, :, 1] = (results_per_image.pred_keypoints[:, :, 1] + 0.5) * roi_h + offset_y

                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results
            
    def _integral_target_generator(self, joints_3d, num_joints, patch_height, patch_width):
        '''
        joint_3d.shape = (n,17,3)
        '''
        num_inst_per_img = len(joints_3d)
        target_weight = torch.ones((num_inst_per_img,num_joints, 2), dtype=torch.float32, device=joints_3d.device)
        target_weight[:, :, 0] = joints_3d[:, :, -1]
        target_weight[:, :, 1] = joints_3d[:, :, -1]

        target_visible = torch.ones((num_inst_per_img, num_joints, 1), dtype=torch.float32, device=joints_3d.device)
        target_visible[:, :, 0] = target_weight[:, :, 0]

        target = torch.zeros((num_inst_per_img, num_joints, 2), dtype=torch.float32, device=joints_3d.device)
        target[:, :, 0] = joints_3d[:, :, 0] / patch_width - 0.5
        target[:, :, 1] = joints_3d[:, :, 1] / patch_height - 0.5

        target_visible[target[:, :, 0] > 0.5] = 0
        target_visible[target[:, :, 0] < -0.5] = 0
        target_visible[target[:, :, 1] > 0.5] = 0
        target_visible[target[:, :, 1] < -0.5] = 0

        # target_visible_weight = target_weight[:, :, :1].clone()
        target_weight[:,:,0] = target_visible[:,:,0]
        target_weight[:,:,1] = target_visible[:,:,0]

        # target = target.reshape((-1))
        # target_weight = target_weight.reshape((-1))
        return target, (target_weight > 0).float() #, target_visible[None], target_visible_weight[None]

    def prepare_targets(self, targets, inp_shape):
        new_targets = []
        for targets_per_image in targets:
            
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            
            kps = targets_per_image.gt_keypoints.tensor
            # normed_kps, kps_vis = self._integral_target_generator(kps, kps.shape[1], w, h)
            target['kps_coord'] = kps[:,:,:-1].to(self.device)
            target['kps_vis'] = kps[:,:,-1:].to(self.device)   
            
            hm_hp = np.zeros((kps.shape[1], inp_shape[-2], inp_shape[-1]), dtype=np.float32)
            stride_ = 4 # 
            for inst in range(kps.shape[0]):
                box = image_size_xyxy * gt_boxes
                h = box[inst, 2].cpu().numpy()
                w = box[inst, 3].cpu().numpy()
                hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                hp_radius = max(0, int(hp_radius)) 
                
                for kp_idx in range(kps.shape[1]):
                    if kps[inst, kp_idx, 2] > 0:
                        pts = kps[inst, kp_idx, :-1] / stride_
                        pt_int = pts.cpu().numpy().astype(np.int32)
                        draw_umich_gaussian(hm_hp[kp_idx], pt_int, hp_radius)
            target['heatmap'] = torch.from_numpy(hm_hp).to(self.device)
            
            new_targets.append(target)

        return new_targets

    def inference(self, box_cls, box_pred, kps_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device).\
                     unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, pose_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, kps_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                pose_pred_per_image = pose_pred_per_image.view(-1, 1, self.num_kps, 3).repeat(1, self.num_classes, 1, 1).view(-1, self.num_kps, 3)
                pose_pred_per_image =  pose_pred_per_image[topk_indices]

                kps_scores = pose_pred_per_image[:,:,2].mean(1)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image * kps_scores 
                result.pred_classes = labels_per_image
                result.pred_keypoints = pose_pred_per_image
                results.append(result)


        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh

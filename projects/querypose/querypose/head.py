# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import math
from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F
 
from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes
import collections
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class Decoder(nn.Module):
    # including stacked box decoder following sparsercnn and pose decode 
    def __init__(self, cfg, roi_input_shape, matcher):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler

        pose_pooler = self._init_pose_pooler(cfg, roi_input_shape)
        self.pose_pooler = pose_pooler

        self.matcher = matcher
        
        # Build heads.
        num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        d_model = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.SparseRCNN.DIM_FEEDFORWARD
        nhead = cfg.MODEL.SparseRCNN.NHEADS
        dropout = cfg.MODEL.SparseRCNN.DROPOUT
        activation = cfg.MODEL.SparseRCNN.ACTIVATION
        num_heads = cfg.MODEL.SparseRCNN.NUM_HEADS
        rcnn_head = RCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation)        
        self.head_series = _get_clones(rcnn_head, num_heads)
        pose_head = Pose_Decoder_Layer(cfg)
        self.pose_head_series = _get_clones(pose_head, num_heads)
        self.return_intermediate = cfg.MODEL.SparseRCNN.DEEP_SUPERVISION
        
        # Init parameters.
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        self.num_classes = num_classes
        if self.use_focal:
            prior_prob = cfg.MODEL.SparseRCNN.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if self.use_focal:
                if p.shape[-1] == self.num_classes:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    @staticmethod
    def _init_pose_pooler(cfg, input_shape):
        #import pudb;pudb.set_trace()
        in_features = cfg.MODEL.QueryPose.ROI_HEADS_IN_FEATURES_POSE
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        pose_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return pose_pooler

    def forward(self, features, init_bboxes, init_features, part_query, **kwargs):

        inter_class_logits = []
        inter_pred_bboxes = []
        inter_pred_pose = []
        inter_pred_sigma = []
        inter_match_list = []


        bs = len(features[0])
        bboxes = init_bboxes
        nr_boxes = init_features.shape[0]
        init_features = init_features[None].repeat(1, bs, 1)
        part_query = part_query.unsqueeze(1) #.repeat(bs*nr_boxes, 1, 1).permute(1,0,2) # num_part, N*nr_boxes, 128
    
        proposal_features = init_features.clone()
        
        for idx, rcnn_head in enumerate(self.head_series):
            # import pudb;pudb.set_trace()
            class_logits, pred_bboxes, proposal_features = rcnn_head(features, bboxes, proposal_features, self.box_pooler)
            bboxes = pred_bboxes.detach()
            box_output_dict = {'pred_logits': class_logits, 'pred_boxes': pred_bboxes}
            if self.training:
                targets = kwargs['gt']
                indices = self.matcher(box_output_dict, targets) # e.g. [(tensor([17, 29]), tensor([0, 1])), (tensor([99]), tensor([0]))] 
            else:
                indices = None
            pred_pose, sigma, proposal_features, part_query = self.pose_head_series[idx](features, proposal_features, bboxes, self.pose_pooler, part_query, 
                                                                                         indices = indices)
    

            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
                inter_pred_pose.append(pred_pose)
                inter_pred_sigma.append(sigma)
                inter_match_list.append(indices)

        if self.return_intermediate:
            
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes), torch.stack(inter_pred_pose), \
                   torch.stack(inter_pred_sigma), inter_match_list

    


class RCNNHead(nn.Module):

    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.d_model = d_model

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(cfg, cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # cls.
        num_cls = cfg.MODEL.SparseRCNN.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.SparseRCNN.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)
        
        # pred.
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        if self.use_focal:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights


    def forward(self, features, bboxes, pro_features, pooler):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """

        N, nr_boxes = bboxes.shape[:2]
        
        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        roi_features = pooler(features, proposal_boxes)            
        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)        

        # self_att.
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)
        
        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))
        
        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features
    

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class DynamicConv(nn.Module):

    def __init__(self, cfg, pooler_resolution):
        super().__init__()

        self.hidden_dim = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.SparseRCNN.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.SparseRCNN.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (num_tokens, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




def _make_stack_3x3_convs(num_convs, in_channels, hidden_channels, out_channels):
    convs = []
    conv_dim = [hidden_channels]*(num_convs-1) + [out_channels]
    for out_channels in conv_dim:
        convs.append(
            nn.Conv2d(in_channels, out_channels, 3, padding=1))
        convs.append(nn.ReLU(True))
        in_channels = out_channels
        
    return nn.Sequential(*convs)


class Spatial_Part_Embedding_Generation(nn.Module):

    def __init__(self, cfg, in_channels, out_channels, scale):
        super().__init__()    
        
        self.num_parts = cfg.MODEL.QueryPose.NUM_PART
        self.scale = scale
        if self.scale:
            self.upsample = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=4 // 2 - 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),)
                # nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=4 // 2 - 1, bias=False),
                # nn.BatchNorm2d(in_channels), 
                # nn.ReLU(inplace=True),)

        self.LSA_conv = nn.Conv2d(in_channels, self.num_parts, 3, padding=1)
        # outputs
        self.fc = nn.Linear(in_channels, out_channels)

        self._init_weights()

    def _init_weights(self):
      
        if self.scale:
            for m in self.upsample.modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
        nn.init.normal_(self.LSA_conv.weight, std=0.01)
        nn.init.constant_(self.LSA_conv.bias, 0.0)
        
        c2_xavier_fill(self.fc)

    def forward(self, features):
        if self.scale:
            features = self.upsample(features)
            #features = F.interpolate(features, scale_factor=2.0, mode="bilinear", align_corners=False)
        local_spatial_attn = self.LSA_conv(features)
    
        B, num_part = local_spatial_attn.shape[:2]
        C = features.size(1)
        normal_local_spatial_attn = local_spatial_attn.view(B, num_part, -1).softmax(-1)
        part_embed = torch.bmm(normal_local_spatial_attn, features.view(B, C, -1).permute(0, 2, 1))
        #part_features = torch.einsum('bnhw,bchw->bnc', pam_prob, features)

        part_embed = part_embed.reshape(B, num_part, -1).permute(1,0,2) # part, B*Nr_box, d_model
     
        part_embed = self.fc(part_embed) 
        
        return part_embed       


class Selective_updater(nn.Module):
    # the user can design the more complex module according to the GRU
    def __init__(self, input_dim, scale_factor = 4):
        super(Selective_updater, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, input_dim//scale_factor, bias=False)
        self.norm_fc1 = nn.LayerNorm(input_dim//scale_factor)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(input_dim//scale_factor, 2 * input_dim, bias=True)

        self.fc3 = nn.Linear(input_dim, input_dim, bias=False)
        self.norm_fc3  = nn.LayerNorm(input_dim)
        self.D = input_dim

    def forward(self, update_feat, input_feat):

        num_p, B = update_feat.shape[:2]
        d  = update_feat + input_feat
        d = F.relu(self.norm_fc1(self.fc1(d)))
        d = self.fc2(d)
        d = d.view(num_p, B, 2, self.D)
        #d = F.softmax(d, 2)
        d = torch.sigmoid(d)
        d1 = update_feat * d[:, :, 0, :]
        d2 = input_feat * d[:, :, 1, :]
        d  = d1 + d2

        #out = self.fc3(d)
        #out = self.norm_fc3(out)
        #out = self.relu(out)
        

        return d #out




class Pose_Decoder_Layer(nn.Module):
    def __init__(self, cfg):
        super(Pose_Decoder_Layer, self).__init__()
        d_model = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.SparseRCNN.DIM_FEEDFORWARD
        dropout = cfg.MODEL.SparseRCNN.DROPOUT
        activation = cfg.MODEL.SparseRCNN.ACTIVATION
        self.light = cfg.MODEL.QueryPose.LIGHT_VERSION 
        self.part_dim = cfg.MODEL.QueryPose.PART_DIM
        self.d_model = d_model
        
        self.roi_size = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        
        parts = {"face":5,"shoulder":2,
                        "left_elbow_wrist":2,"right_elbow_wrist":2,
                        "hip":2,
                        "left_knee_ankle":2,
                        "right_knee_ankle":2}

        # parts = {"face":5,
        #         "left_shoulder_elbow_wrist":3,"right_shoulder_elbow_wrist":3,
        #         "left_hip_knee_ankle":3,
        #         "right_hip_knee_ankle":3}

        self.parts = collections.OrderedDict(parts)
        self.num_parts = len(self.parts)
        assert self.num_parts == cfg.MODEL.QueryPose.NUM_PART
        self.num_joints = cfg.MODEL.QueryPose.NUM_KPS 
        num_convs = cfg.MODEL.QueryPose.NUM_CONVS
        
        self.inst_convs = _make_stack_3x3_convs(num_convs, self.d_model, self.d_model*2, self.d_model)
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        

        self.activation = _get_activation_fn(activation)
        if not self.light:
            self.pose_decoder_inst_interact = DynamicConv(cfg, self.roi_size)

            self.inst_norm = nn.LayerNorm(d_model)
            self.inst_dropout = nn.Dropout(dropout)

    ########################## feed-forward network
            self.norm2 = nn.LayerNorm(d_model)    
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
            self.linear1 = nn.Linear(d_model, dim_feedforward)      
            self.linear2 = nn.Linear(dim_feedforward, d_model)
    ############################

        scale = True
        self.part_embed_generator = Spatial_Part_Embedding_Generation(cfg, self.d_model, self.part_dim, scale)
    
        self.updater = Selective_updater(self.part_dim)

        ########################## mhsa across part query and feed-forward network
        nhead = cfg.MODEL.SparseRCNN.NHEADS
        self.kps_self_attn = nn.MultiheadAttention(self.part_dim, nhead, dropout=dropout)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn_norm = nn.LayerNorm(self.part_dim)

        self.norm2_l = nn.LayerNorm(self.part_dim)    
        self.dropout2_l = nn.Dropout(dropout)
        self.dropout3_l = nn.Dropout(dropout)
        self.linear1_l = nn.Linear(self.part_dim, dim_feedforward)      
        self.linear2_l = nn.Linear(dim_feedforward, self.part_dim)



        self.fc_coord = nn.ModuleList()
        self.fc_sigma = nn.ModuleList()
        self.residual_filter = nn.ModuleList()
      
        for part in self.parts.keys():
            self.residual_filter.append(nn.Linear(self.d_model, self.part_dim)) 
            self.fc_coord.append(Linear(self.part_dim, 2 * self.parts[part]))
            self.fc_sigma.append(Linear(self.part_dim, 2 * self.parts[part], norm=False))

     

    def post_process(self, res_dict):
        final_result = [res_dict["face"],res_dict["shoulder"],
                        res_dict["left_elbow_wrist"][:,:2],res_dict["right_elbow_wrist"][:,:2],
                        res_dict["left_elbow_wrist"][:,2:],res_dict["right_elbow_wrist"][:,2:],
                        res_dict["hip"],
                        res_dict["left_knee_ankle"][:,:2],res_dict["right_knee_ankle"][:,:2],
                        res_dict["left_knee_ankle"][:,2:],res_dict["right_knee_ankle"][:,2:]]
        final_result = torch.cat(final_result, dim=1)
        return final_result 

    # def post_process(self,res_dict):
    #     final_result = [res_dict["face"],
    #                     res_dict["left_shoulder_elbow_wrist"][:,:2],res_dict["right_shoulder_elbow_wrist"][:,:2],
    #                     res_dict["left_shoulder_elbow_wrist"][:,2:4],res_dict["right_shoulder_elbow_wrist"][:,2:4],
    #                     res_dict["left_shoulder_elbow_wrist"][:,4:],res_dict["right_shoulder_elbow_wrist"][:,4:],
    #                     res_dict["left_hip_knee_ankle"][:,:2],res_dict["right_hip_knee_ankle"][:,:2],
    #                     res_dict["left_hip_knee_ankle"][:,2:4],res_dict["right_hip_knee_ankle"][:,2:4],
    #                     res_dict["left_hip_knee_ankle"][:,4:],res_dict["right_hip_knee_ankle"][:,4:]]
    #     final_result = torch.cat(final_result, dim=1)
    #     return final_result 

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, features, inst_query, bboxes, pose_pooler, part_query, **kwargs):
        # roi_feature only using p2.
        features = [features[0]]
        N, nr_boxes = bboxes.shape[:2]
        all_inst = N * nr_boxes
        all_inst_query = inst_query

        if self.training:
            indices = kwargs['indices']
            proposal_boxes = list()
            valid_query_ind = []
            for i, indice in enumerate(indices):
                pred_ind, target_ind = indice #[(tensor([17, 29]), tensor([0, 1])), (tensor([99]), tensor([0]))] 
                proposal_boxes.append(Boxes(bboxes[i][pred_ind]))
                valid_query_ind.append(pred_ind + nr_boxes * i)
            valid_query_ind = torch.cat(valid_query_ind, dim = 0)
            inst_query = inst_query.squeeze(0)[valid_query_ind].unsqueeze(0) # 1, num_valid_inst, 256
            all_inst = inst_query.shape[1]
            if all_inst == 0:
                # import pudb;pudb.set_trace()
                return torch.zeros((all_inst, self.num_joints, 3), device=all_inst_query.device), torch.zeros((all_inst, self.num_joints, 2), device=all_inst_query.device), all_inst_query, part_query

        else:
            proposal_boxes = list()
            for b in range(N):
                proposal_boxes.append(Boxes(bboxes[b]))


        roi_features = pose_pooler(features, proposal_boxes) 

        pose_roi_features = roi_features.view(all_inst, self.d_model, self.roi_size, self.roi_size)
        pose_roi_features = self.inst_convs(pose_roi_features)
        
         
        if not self.light:
            inst_roi_features = pose_roi_features.clone().view(all_inst, self.d_model, -1).permute(2, 0, 1)
            inst_query = inst_query.reshape(1, all_inst, self.d_model)
            inst_feat = self.pose_decoder_inst_interact(inst_query, inst_roi_features)
            inst_query1 = inst_query + self.inst_dropout(inst_feat)
            inst_query1 = self.inst_norm(inst_query1)

            # FFN
            inst_query2 = self.linear2(self.dropout2(self.activation(self.linear1(inst_query1))))
            inst_query2 = inst_query1 + self.dropout3(inst_query2)
            inst_query = self.norm2(inst_query2)
        ##########################################################################################################
        
        
        residual_embed_list = []
        for i in range(self.num_parts):
            spera_residual_embed = self.residual_filter[i](inst_query)
            residual_embed_list.append(spera_residual_embed)
        residual_embed = torch.cat(residual_embed_list, dim=0)  # 7, N * nr_boxes, 128
        
        part_embed = self.part_embed_generator(pose_roi_features) + residual_embed
        if part_query is not None:
            part_query = self.updater(part_embed, part_query)
     
        # import pudb;pudb.set_trace()
        part_query1 = self.kps_self_attn(part_query, part_query, value = part_query)[0]
        part_query = part_query + self.self_attn_dropout(part_query1)
        part_query = self.self_attn_norm(part_query)
        
        
        # FFN
        part_query2 = self.linear2_l(self.dropout2_l(self.activation(self.linear1_l(part_query))))
        part_query = part_query + self.dropout3_l(part_query2)
        part_query = self.norm2_l(part_query)
        
        out_coord = collections.OrderedDict()
        out_sigma = collections.OrderedDict()
        for j, head in enumerate(list(self.parts.keys())):
            out_coord[head]= self.fc_coord[j](part_query[j]).squeeze(0).view(all_inst, -1)
            out_sigma[head]= self.fc_sigma[j](part_query[j]).squeeze(0).view(all_inst, -1)

        out_coord = self.post_process(out_coord)
        out_sigma = self.post_process(out_sigma)

        # out_coord = []
        # out_sigma = []
        # for j in range(self.num_joints):
        #     out_coord.append(self.fc_coord[j](kps_embed[j]).view(N*nr_boxes,1, 2))
        #     out_sigma.append(self.fc_sigma[j](kps_embed[j]).view(N*nr_boxes,1, 2))
        # out_coord = torch.cat(out_coord, dim=1)
        # out_sigma = torch.cat(out_sigma, dim=1)

        
        # (B, N, 2)
        pred_kps = out_coord.reshape(all_inst, self.num_joints, 2)
        sigma = out_sigma.reshape(all_inst, self.num_joints, -1).sigmoid()
        scores = 1 - sigma

        scores = torch.mean(scores, dim=2, keepdim=True)
        pred_pose = torch.cat([pred_kps, scores], dim = 2)
        if self.training:
            all_inst_query[0, valid_query_ind] = inst_query[0]
        return pred_pose.view(all_inst, self.num_joints, 3), sigma.view(all_inst, self.num_joints, 2), all_inst_query, part_query



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

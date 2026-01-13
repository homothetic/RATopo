import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv.cnn import Linear, bias_init_with_prob, build_activation_layer
from mmcv.cnn.bricks.transformer import build_feedforward_network, build_transformer_layer
from mmcv.runner import auto_fp16, force_fp32
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.cnn import xavier_init
from scipy.optimize import linear_sum_assignment
from ..modules.position_embed import gen_sineembed_for_position
from ..modules.ranking_losses import APLoss
from ...core.lane.util import fix_pts_interpolate


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@HEADS.register_module()
class TopoLLHead(nn.Module):
    def __init__(self,
                 in_channels_o1,
                 in_channels_o2=None,
                 shared_param=False,
                 loss_rel=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25),
                 dis_function='l2',
                 mapping_function='learnable',
                 learnable_alpha=0.2,
                 learnable_lambda=2.0,
                 divide_sigma=True,
                 is_detach=False,
                 pts_dim=3,
                 ranking_loss=0,
                ):
        super().__init__()

        self.shared_param = False
        self.MLP_o1 = MLP(in_channels_o1, in_channels_o1, 256, 3)
        self.MLP_o2 = MLP(in_channels_o2, in_channels_o2, 256, 3)
        self.loss_rel = build_loss(loss_rel)
        self.is_detach = is_detach
        self.pts_dim = pts_dim
        self.ranking_loss = ranking_loss
        
        self.dis_function = dis_function
        self.mapping_function = mapping_function
        if self.mapping_function == 'learnable':
            self._alpha = nn.Parameter(torch.tensor(learnable_alpha))
            self._lambda = nn.Parameter(torch.tensor(learnable_lambda))
            self.divide_sigma = divide_sigma
        elif self.mapping_function == 'mlp':
            self.mlp_dist = MLP(1, 16, 1, 3) # 1 -> 16 -> 16 -> 1

        self._lambda_1 = nn.Parameter(torch.tensor(1.0))
        self._lambda_2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, o1_feats, o2_feats, o1_pos, o2_pos, **kwargs):
        # breakpoint()
        # feats: [B, num_query, num_embedding]
        # pos: [B, num_query, num_pts * pts_dim]
        
        # Lane Similarity Topology
        o1_feat = o1_feats.clone()
        o2_feat = o2_feats.clone()
        o1_embeds = self.MLP_o1(o1_feat)
        o2_embeds = self.MLP_o2(o2_feat)

        sim = torch.matmul(o1_embeds, torch.transpose(o2_embeds, 1, 2)) # [B, num_query, num_query]
        sim = sim.unsqueeze(-1) # [B, num_query, num_query, 1]
        G_sim = sim.sigmoid()    

        # Lane Geometric Distance Matrix
        o1_pos = o1_pos.clone()
        o2_pos = o2_pos.clone()
        if self.is_detach:
            G_sim = G_sim.detach()
            o1_pos = o1_pos.detach()
            o2_pos = o2_pos.detach()
        o1 = o1_pos[..., - self.pts_dim : ].unsqueeze(2).repeat(1, 1, o2_pos.size(1), 1)
        o2 = o2_pos[..., : self.pts_dim].unsqueeze(1).repeat(1, o1_pos.size(1), 1, 1)
        if self.dis_function == 'l2':
            geo_dist = torch.sqrt(torch.pow(o1 - o2, 2).sum(-1)) # [B, num_query, num_query]
        elif self.dis_function == 'l1':
            geo_dist = torch.abs(o1 - o2).sum(-1) # [B, num_query, num_query]
        geo_dist = geo_dist.unsqueeze(-1) # [B, num_query, num_query, 1]

        # Distance to Topology Mapping Function
        if self.mapping_function == 'learnable':
            if self.divide_sigma:
                sigma = torch.std(geo_dist.view(geo_dist.size(0), -1), dim=-1) # [B]
                sigma = sigma[:, None, None, None].repeat(1, geo_dist.size(1), geo_dist.size(2), geo_dist.size(3))
                G_dis = torch.exp(- torch.pow(geo_dist, self._alpha) / (self._lambda * sigma))
            else:
                G_dis = torch.exp(- torch.pow(geo_dist, self._alpha) / self._lambda)
        elif self.mapping_function == 'gaussian':
            G_dis = torch.exp(- torch.pow(geo_dist, 2.0) / 2.0)
        elif self.mapping_function == 'mlp':
            G_dis = self.mlp_dist(geo_dist).sigmoid()
        else:
            raise NotImplementedError
        topo_mask = 1 - torch.eye(G_dis.size(1), dtype=G_dis.dtype, device=G_dis.device)
        G_dis = G_dis * topo_mask[None, :, :, None]

        G_topo = (self._lambda_1 * G_sim + self._lambda_2 * G_dis) / (self._lambda_1 + self._lambda_2)
        return sim, G_topo

    def loss(self, hs, rel_preds, pos_preds, gt_adjs, gt_lanes, o1_assign_results, o2_assign_results):
        B, num_query_o1, num_query_o2, _ = rel_preds.size()
        o1_assign = o1_assign_results
        o1_pos_inds = o1_assign['pos_inds']
        o1_pos_assigned_gt_inds = o1_assign['pos_assigned_gt_inds']

        if self.shared_param:
            o2_assign = o1_assign
            o2_pos_inds = o1_pos_inds
            o2_pos_assigned_gt_inds = o1_pos_assigned_gt_inds
        else:
            o2_assign = o2_assign_results
            o2_pos_inds = o2_assign['pos_inds']
            o2_pos_assigned_gt_inds = o2_assign['pos_assigned_gt_inds']

        losses_rel = 0
        # losses_ll_l1 = 0
        losses_rank = 0
        losses_rank_norm = 0
        for i in range(B):
            gt_adj = gt_adjs[i]
            target = torch.zeros_like(rel_preds[i].squeeze(-1), dtype=gt_adj.dtype, device=rel_preds.device)
            xs = o1_pos_inds[i].unsqueeze(-1).repeat(1, o2_pos_inds[i].size(0))
            ys = o2_pos_inds[i].unsqueeze(0).repeat(o1_pos_inds[i].size(0), 1)
            target[xs, ys] = gt_adj[o1_pos_assigned_gt_inds[i]][:, o2_pos_assigned_gt_inds[i]]
            xs_new = o1_pos_inds[i]
            ys_new = o2_pos_inds[i]
            
            # copy for ll l1 loss
            target_copy = target

            if self.ranking_loss:
                # breakpoint()
                target = target[xs_new][:, ys_new].long()
                rel_pred = rel_preds[i][xs_new][:, ys_new]
                loss_rel = self.loss_rel(rel_pred.view(-1, 1), 1 - target.view(-1))
                if self.ranking_loss == 1:
                    if target.sum() > 0:
                        aploss_rel_pred = rel_pred.reshape(-1)
                        aploss_target = target.reshape(-1)
                        losses_rank += APLoss.apply(aploss_rel_pred, aploss_target)
                        losses_rank_norm += 1
                elif self.ranking_loss == 2:
                    for row in range(len(target)):
                        aploss_rel_pred = rel_pred[row].reshape(-1)
                        aploss_target = target[row].reshape(-1)
                        if aploss_target.sum() > 0:
                            losses_rank += APLoss.apply(aploss_rel_pred, aploss_target)
                            losses_rank_norm += 1
                    for col in range(target.size(1)):
                        aploss_rel_pred = rel_pred[:, col].reshape(-1)
                        aploss_target = target[:, col].reshape(-1)
                        if aploss_target.sum() > 0:
                            losses_rank += APLoss.apply(aploss_rel_pred, aploss_target)
                            losses_rank_norm += 1
            else:
                # lane gt num * lane gt num
                target = 1 - target[xs_new][:, ys_new].view(-1).long()
                rel_pred = rel_preds[i][xs_new][:, ys_new].view(-1, 1)
                # target = 1 - target.view(-1).long()
                # rel_pred = rel_preds[i].view(-1, 1)
                loss_rel = self.loss_rel(rel_pred, target)

            if digit_version(TORCH_VERSION) >= digit_version('1.8'):
                loss_rel = torch.nan_to_num(loss_rel)
            losses_rel += loss_rel

        if losses_rank_norm > 0:
            if digit_version(TORCH_VERSION) >= digit_version('1.8'):
                losses_rank = torch.nan_to_num(losses_rank)
            # return dict(loss_rel=losses_rel / B, loss_rank=losses_rank / losses_rank_norm)
            return dict(loss_rel=losses_rel / B + losses_rank / losses_rank_norm)
        return dict(loss_rel=losses_rel / B)


@HEADS.register_module()
class TopoLTHead(nn.Module):
    def __init__(self,
                 in_channels_o1,
                 in_channels_o2=None,
                 shared_param=False,
                 loss_rel=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25),
                 with_rel_loss=True,
                 add_pos=False,
                 pos_dimension=9,
                 is_detach=False,
                 ranking_loss=0,):
        super().__init__()

        self.MLP_o1 = MLP(in_channels_o1, in_channels_o1, 128, 3)
        self.shared_param = shared_param
        if shared_param:
            self.MLP_o2 = self.MLP_o1
        else:
            self.MLP_o2 = MLP(in_channels_o2, in_channels_o2, 128, 3)
        self.classifier = MLP(256, 256, 1, 3)
        self.loss_rel = build_loss(loss_rel)
        self.with_rel_loss = with_rel_loss
        self.ranking_loss = ranking_loss

        self.add_pos = add_pos
        if self.add_pos:
            self.pos_embed = MLP(pos_dimension, 128, 128, 3)
        self.is_detach = is_detach

    def forward(self, o1_feats, o2_feats, img_metas=None):
        # feats: [B, num_query, num_embedding]
        o1_embeds = o1_feats.clone()
        o2_embeds = o2_feats.clone()

        if self.is_detach:
            o1_embeds = o1_embeds.detach()
            o2_embeds = o2_embeds.detach()

        o1_embeds = self.MLP_o1(o1_embeds)
        o2_embeds = self.MLP_o2(o2_embeds)

        if self.add_pos:
            lidar2img = torch.tensor(img_metas[0]['lidar2img'][0][:3, :3].flatten()).to(torch.float32).to(o1_embeds.device)
            o1_embeds = self.pos_embed(lidar2img) + o1_embeds

        num_query_o1 = o1_embeds.size(1)
        num_query_o2 = o2_embeds.size(1)
        o1_tensor = o1_embeds.unsqueeze(2).repeat(1, 1, num_query_o2, 1)
        o2_tensor = o2_embeds.unsqueeze(1).repeat(1, num_query_o1, 1, 1)

        relationship_tensor = torch.cat([o1_tensor, o2_tensor], dim=-1)
        relationship_pred = self.classifier(relationship_tensor)

        return relationship_pred

    def loss(self, rel_preds, gt_adjs, o1_assign_results, o2_assign_results):
        B, num_query_o1, num_query_o2, _ = rel_preds.size()
        o1_assign = o1_assign_results
        o1_pos_inds = o1_assign['pos_inds']
        o1_pos_assigned_gt_inds = o1_assign['pos_assigned_gt_inds']

        if self.shared_param:
            o2_assign = o1_assign
            o2_pos_inds = o1_pos_inds
            o2_pos_assigned_gt_inds = o1_pos_assigned_gt_inds
        else:
            o2_assign = o2_assign_results
            o2_pos_inds = o2_assign['pos_inds']
            o2_pos_assigned_gt_inds = o2_assign['pos_assigned_gt_inds']

        losses_rel = 0
        losses_rank = 0
        losses_rank_norm = 0
        for i in range(B):
            gt_adj = gt_adjs[i]
            target = torch.zeros_like(rel_preds[i].squeeze(-1), dtype=gt_adj.dtype, device=rel_preds.device)
            xs = o1_pos_inds[i].unsqueeze(-1).repeat(1, o2_pos_inds[i].size(0))
            ys = o2_pos_inds[i].unsqueeze(0).repeat(o1_pos_inds[i].size(0), 1)
            target[xs, ys] = gt_adj[o1_pos_assigned_gt_inds[i]][:, o2_pos_assigned_gt_inds[i]]
            xs_new = o1_pos_inds[i]
            ys_new = o2_pos_inds[i]
            
            if self.ranking_loss:
                # breakpoint()
                if not self.with_rel_loss:
                    # target = 1 - target.view(-1).long()
                    # rel_pred = rel_preds[i].view(-1, 1)
                    target = target.long()
                    rel_pred = rel_preds[i]
                else:
                    # target = 1 - target[xs_new].view(-1).long()
                    # rel_pred = rel_preds[i][xs_new].view(-1, 1)
                    target = target[xs_new].long()
                    rel_pred = rel_preds[i][xs_new]
                loss_rel = self.loss_rel(rel_pred.view(-1, 1), 1 - target.view(-1))
                if self.ranking_loss == 1:
                    if target.sum() > 0:
                        aploss_rel_pred = rel_pred.reshape(-1)
                        aploss_target = target.reshape(-1)
                        losses_rank += APLoss.apply(aploss_rel_pred, aploss_target)
                        losses_rank_norm += 1
                elif self.ranking_loss == 2:
                    for row in range(len(target)):
                        aploss_rel_pred = rel_pred[row].reshape(-1)
                        aploss_target = target[row].reshape(-1)
                        if aploss_target.sum() > 0:
                            losses_rank += APLoss.apply(aploss_rel_pred, aploss_target)
                            losses_rank_norm += 1
            else:
                # lane gt num * traffic pred num
                if self.with_rel_loss:
                    target = 1 - target[xs_new].view(-1).long()
                    rel_pred = rel_preds[i][xs_new].view(-1, 1)
                else:
                    target = 1 - target.view(-1).long()
                    rel_pred = rel_preds[i].view(-1, 1)
                loss_rel = self.loss_rel(rel_pred, target)

            if digit_version(TORCH_VERSION) >= digit_version('1.8'):
                loss_rel = torch.nan_to_num(loss_rel)
            losses_rel += loss_rel

        if losses_rank_norm > 0:
            if digit_version(TORCH_VERSION) >= digit_version('1.8'):
                losses_rank = torch.nan_to_num(losses_rank)
            # return dict(loss_rel=losses_rel / B, loss_rank=losses_rank / losses_rank_norm)
            return dict(loss_rel=losses_rel / B + losses_rank / losses_rank_norm)
        return dict(loss_rel=losses_rel / B)

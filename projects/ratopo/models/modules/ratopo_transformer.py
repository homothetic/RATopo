import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner import auto_fp16, force_fp32
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER

from projects.bevformer.modules.decoder import CustomMSDeformableAttention
from projects.bevformer.modules.spatial_cross_attention import \
    MSDeformableAttention3D
from projects.bevformer.modules.temporal_self_attention import \
    TemporalSelfAttention


@TRANSFORMER.register_module()
class RATopoTransformer(BaseModule):

    def __init__(self,
                 decoder=None,
                 embed_dims=256,
                 pts_dim=3,
                 **kwargs):
        super(RATopoTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.fp16_enabled = False
        self.pts_dim = pts_dim
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.reference_points = nn.Linear(self.embed_dims, self.pts_dim)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_embed,
                object_query_embed,
                bev_h,
                bev_w,
                cls_branches=None,
                reg_branches=None,
                lclc_branches=None,
                lcte_branches=None,
                te_feats=None,
                te_cls_scores=None,
                **kwargs):
        # breakpoint()
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)

        if 'group_embed' in kwargs and kwargs['group_embed'] is not None:
            query_pos = query_pos+kwargs['group_embed'].unsqueeze(0).expand(bs, -1, -1)


        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)
        inter_states, inter_references, inter_cls, inter_reg, inter_lclc_rel, inter_lcte_rel = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            cls_branches=cls_branches,
            reg_branches=reg_branches,
            lclc_branches=lclc_branches,
            lcte_branches=lcte_branches,
            te_feats=te_feats,
            te_cls_scores=te_cls_scores,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return inter_states, init_reference_out, inter_references_out, inter_cls, inter_reg, inter_lclc_rel, inter_lcte_rel

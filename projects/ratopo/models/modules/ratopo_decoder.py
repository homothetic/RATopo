import copy
import warnings
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import mmcv
from mmcv.cnn import Linear, build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout 
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER, FEEDFORWARD_NETWORK,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import BaseTransformerLayer, TransformerLayerSequence
from mmcv.cnn.bricks.transformer import build_transformer_layer
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.cnn import xavier_init
from scipy.optimize import linear_sum_assignment
from .position_embed import gen_sineembed_for_position
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


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class RATopoDecoder(BaseModule): # TransformerLayerSequence

    def __init__(self, 
                 *args, 
                 transformerlayers=None, 
                 num_layers=None, 
                 init_cfg=None,
                 inquiry_heads_num=1,
                 return_intermediate=False, 
                 pc_range=None, 
                 pts_dim=3, 
                 topo_head='toponet', 
                 num_lanes_one2one=300,
                 lanes_group=None,
                 use_attn_mask=False,
                 with_box_refine=False,
                 with_multi_point=False,
                 **kwargs):
        
        super().__init__(init_cfg)
        import copy
        if isinstance(transformerlayers, dict):
            transformerlayers = [copy.deepcopy(transformerlayers) for _ in range(num_layers)]
        else:
            assert isinstance(transformerlayers, list) and len(transformerlayers) == num_layers
        self.num_layers = num_layers
        layers = nn.ModuleList()
        for i in range(num_layers):
            layers.append(build_transformer_layer(transformerlayers[i]))
        self.embed_dims = layers[0].embed_dims
        self.pre_norm = layers[0].pre_norm

        # multi inquiry
        # breakpoint()
        self.inquiry_heads_num = inquiry_heads_num
        self.mi_layers = nn.ModuleList([copy.deepcopy(layers) for i in range(self.inquiry_heads_num)])
        self.fusion_layer = Linear(self.inquiry_heads_num * self.embed_dims, self.embed_dims)

        self.return_intermediate = return_intermediate
        self.pc_range = pc_range
        self.pts_dim = pts_dim
        self.topo_head = topo_head
        self.num_lanes_one2one = num_lanes_one2one
        self.lanes_group = lanes_group
        self.use_attn_mask = use_attn_mask
        if self.use_attn_mask:
            if self.use_attn_mask == 2:
                self.attn_mlp = MLP(2, 16, 8, 3)
            elif self.use_attn_mask == 3:
                import copy
                def _get_clones(module, N):
                    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
                _score_embedding = MLP(1, 16, 8, 2)
                self.attn_mlp = _get_clones(_score_embedding, 5)
            else:
                self.attn_mlp = MLP(1, 16, 8, 2)
        self.with_box_refine = with_box_refine
        self.with_multi_point = with_multi_point
        self.fp16_enabled = False

    def forward(self,
                query,
                *args,
                reference_points=None,
                cls_branches=None,
                reg_branches=None,
                lclc_branches=None,
                lcte_branches=None,
                key_padding_mask=None,
                te_feats=None,
                te_cls_scores=None,
                **kwargs):

        output = query
        intermediate = []
        intermediate_reference_points = []
        intermediate_cls = []
        intermediate_reg = []
        intermediate_lclc_rel = []
        intermediate_lcte_rel = []
        num_query, bs = query.size(0), query.size(1)
        num_te_query = te_feats.size(2)

        # breakpoint()
        prev_lclc_adj = torch.zeros((self.inquiry_heads_num, bs, 2, self.num_lanes_one2one, self.num_lanes_one2one),
                                  dtype=query.dtype, device=query.device)
        prev_lcte_adj = torch.zeros((self.inquiry_heads_num, bs, 2, self.num_lanes_one2one, num_te_query),
                                  dtype=query.dtype, device=query.device)
        self_attn_masks = None
        
        for lid in range(self.num_layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2) # BS NUM_QUERY NUM_LEVEL 2
            
            # breakpoint()
            mi_output = []
            mi_output_o2m = []
            for mi_idx in range(self.inquiry_heads_num):
                tmp_output, tmp_output_o2m = self.mi_layers[mi_idx][lid](
                    output,
                    *args,
                    reference_points=reference_points_input,
                    key_padding_mask=key_padding_mask,
                    te_query=te_feats[lid],
                    te_cls_scores=te_cls_scores[lid],
                    lclc_adj=prev_lclc_adj[mi_idx],
                    lcte_adj=prev_lcte_adj[mi_idx],
                    attn_masks=self_attn_masks,
                    **kwargs)
                mi_output.append(tmp_output)
                mi_output_o2m.append(tmp_output_o2m)

            # breakpoint()
            # o2o
            output = self.fusion_layer(torch.cat(mi_output, dim=-1))
            output = output.permute(1, 0, 2)

            assert cls_branches is not None and reg_branches is not None
            outputs_class = cls_branches[lid](output)
            tmp = reg_branches[lid](output)

            reference = reference_points.clone()
            reference = inverse_sigmoid(reference)
            assert reference.shape[-1] == self.pts_dim
            
            bs, num_query, _ = tmp.shape
            tmp = tmp.view(bs, num_query, -1, self.pts_dim)
            tmp = tmp + reference.unsqueeze(2)
            tmp = tmp.sigmoid()

            coord = tmp.clone()
            coord[..., 0] = coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            coord[..., 1] = coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            if self.pts_dim == 3:
                coord[..., 2] = coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outputs_coord = coord.view(bs, num_query, -1).contiguous()

            lclc_rel_out, G_topo = lclc_branches[lid](output, output, outputs_coord, outputs_coord) # TopoLLHeadGeoDistRelLossV1
            lclc_rel_out = torch.stack(lclc_rel_out.split(bs, dim=0), dim=1) # (b, k, l, l, 1), without sigmoid
            G_topo = torch.stack(G_topo.split(bs, dim=0), dim=1) # (b, k, l, l, 1), with sigmoid

            prev_lclc_adj = G_topo.clone() # .detach()
            prev_lclc_adj = prev_lclc_adj.squeeze(-1) # .sigmoid()

            lcte_rel_out = lcte_branches[lid](output, te_feats[lid].repeat(output.size(0) // bs, 1, 1), kwargs['img_metas']) # TopoLTHead
            lcte_rel_out = torch.stack(lcte_rel_out.split(bs, dim=0), dim=1) # (b, k, l, t, 1), without sigmoid

            prev_lcte_adj = lcte_rel_out.clone().detach()
            prev_lcte_adj = prev_lcte_adj.squeeze(-1).sigmoid()

            output = output.permute(1, 0, 2)

            # breakpoint()
            # o2m
            outputs_class_o2m_list = []
            outputs_coord_o2m_list = []
            lclc_rel_out_o2m_list = []
            lcte_rel_out_o2m_list = []
            prev_lclc_adj_o2m_list = []
            prev_lcte_adj_o2m_list = []
            for output_o2m in mi_output_o2m:
                output_o2m = output_o2m.permute(1, 0, 2)

                assert cls_branches is not None and reg_branches is not None
                outputs_class_o2m = cls_branches[lid](output_o2m)
                tmp = reg_branches[lid](output_o2m)

                reference = reference_points.clone()
                reference = inverse_sigmoid(reference)
                assert reference.shape[-1] == self.pts_dim
                
                bs, num_query, _ = tmp.shape
                tmp = tmp.view(bs, num_query, -1, self.pts_dim)
                tmp = tmp + reference.unsqueeze(2)
                tmp = tmp.sigmoid()

                coord = tmp.clone()
                coord[..., 0] = coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
                coord[..., 1] = coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
                if self.pts_dim == 3:
                    coord[..., 2] = coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
                outputs_coord_o2m = coord.view(bs, num_query, -1).contiguous()

                outputs_class_o2m_list.append(outputs_class_o2m)
                outputs_coord_o2m_list.append(outputs_coord_o2m)

                lclc_rel_out_o2m, G_topo_o2m = lclc_branches[lid](output_o2m, output_o2m, outputs_coord_o2m, outputs_coord_o2m) # TopoLLHeadGeoDistRelLossV1
                lclc_rel_out_o2m = torch.stack(lclc_rel_out_o2m.split(bs, dim=0), dim=1) # (b, k, l, l, 1), without sigmoid
                G_topo_o2m = torch.stack(G_topo_o2m.split(bs, dim=0), dim=1) # (b, k, l, l, 1), with sigmoid

                prev_lclc_adj_o2m = G_topo_o2m.clone() # .detach()
                prev_lclc_adj_o2m = prev_lclc_adj_o2m.squeeze(-1) # .sigmoid()

                lcte_rel_out_o2m = lcte_branches[lid](output_o2m, te_feats[lid].repeat(output_o2m.size(0) // bs, 1, 1), kwargs['img_metas']) # TopoLTHead
                lcte_rel_out_o2m = torch.stack(lcte_rel_out_o2m.split(bs, dim=0), dim=1) # (b, k, l, t, 1), without sigmoid

                prev_lcte_adj_o2m = lcte_rel_out_o2m.clone().detach()
                prev_lcte_adj_o2m = prev_lcte_adj_o2m.squeeze(-1).sigmoid()

                lclc_rel_out_o2m_list.append(lclc_rel_out_o2m)
                lcte_rel_out_o2m_list.append(lcte_rel_out_o2m)
                prev_lclc_adj_o2m_list.append(prev_lclc_adj_o2m)
                prev_lcte_adj_o2m_list.append(prev_lcte_adj_o2m)

                # breakpoint() # check whether need to permute
                # output_o2m = output_o2m.permute(1, 0, 2)

            # breakpoint()
            prev_lclc_adj_list = []
            prev_lcte_adj_list = []
            for mi_idx in range(self.inquiry_heads_num):
                prev_lclc_adj_list.append(torch.cat((prev_lclc_adj, prev_lclc_adj_o2m_list[mi_idx]), dim=1))
                prev_lcte_adj_list.append(torch.cat((prev_lcte_adj, prev_lcte_adj_o2m_list[mi_idx]), dim=1))
            prev_lclc_adj = torch.stack(prev_lclc_adj_list, dim=0)
            prev_lcte_adj = torch.stack(prev_lcte_adj_list, dim=0)
            
            if self.training:
                outputs_class = torch.cat((outputs_class, torch.cat(outputs_class_o2m_list, dim=1)), dim=1)
                outputs_coord = torch.cat((outputs_coord, torch.cat(outputs_coord_o2m_list, dim=1)), dim=1)
                lclc_rel_out = torch.cat((lclc_rel_out, torch.cat(lclc_rel_out_o2m_list, dim=1)), dim=1)
                lcte_rel_out = torch.cat((lcte_rel_out, torch.cat(lcte_rel_out_o2m_list, dim=1)), dim=1)

                if self.return_intermediate:
                    intermediate.append(torch.cat((output, torch.cat(mi_output_o2m, dim=0)), dim=0))
                    intermediate_reference_points.append(reference_points)
                    intermediate_cls.append(outputs_class)
                    intermediate_reg.append(outputs_coord)
                    intermediate_lclc_rel.append(lclc_rel_out)
                    intermediate_lcte_rel.append(lcte_rel_out)
            else:
                lclc_rel_out = G_topo[:, 0].squeeze(-1) # .sigmoid()
                lcte_rel_out = lcte_rel_out[:, 0].squeeze(-1).sigmoid()

                if self.return_intermediate:
                    intermediate.append(output)
                    intermediate_reference_points.append(reference_points)
                    intermediate_cls.append(outputs_class)
                    intermediate_reg.append(outputs_coord)
                    intermediate_lclc_rel.append(lclc_rel_out)
                    intermediate_lcte_rel.append(lcte_rel_out)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points), torch.stack(
                intermediate_cls), torch.stack(
                intermediate_reg), torch.stack(
                intermediate_lclc_rel), torch.stack(
                intermediate_lcte_rel)

        return output, reference_points, outputs_class, outputs_coord, lclc_rel_out, lcte_rel_out


@TRANSFORMER_LAYER.register_module()
class RATopoDecoderLayer(BaseTransformerLayer):
    
    def __init__(self,
                 attn_cfgs,
                 ffn_cfgs,
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 num_lanes_one2one=300,
                 lanes_group=None,
                 save_o2m=1,
                 **kwargs):
        super(RATopoDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            **kwargs)
        # assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        self.num_lanes_one2one = num_lanes_one2one
        self.lanes_group = lanes_group
        self.save_o2m = save_o2m
    
    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                te_query=None,
                te_cls_scores=None,
                lclc_adj=None,
                lcte_adj=None,
                **kwargs):

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer_idx, layer in enumerate(self.operation_order):
            # breakpoint()
            if layer == 'self_attn':
                bs = query.size(1)
                if len(query) > self.num_lanes_one2one:
                    query = torch.cat(query.split(self.num_lanes_one2one, dim=0), dim=1)
                    identity = torch.cat(identity.split(self.num_lanes_one2one, dim=0), dim=1)
                    query_pos = torch.cat(query_pos.split(self.num_lanes_one2one, dim=0), dim=1)
                    
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs)
                    
                    query = torch.cat(query.split(bs, dim=1), dim=0)
                    identity = torch.cat(identity.split(bs, dim=1), dim=0)
                    query_pos = torch.cat(query_pos.split(bs, dim=1), dim=0)
                else:
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                if norm_index == 1:
                    query_o2m = self.norms[norm_index](query_o2m)
                    norm_index += 1

                    if self.save_o2m == 1:
                        query = query_o2m.clone() # aux ffn in
                    # elif self.save_o2m == 2:
                    #     query = query # aux ffn out
                    elif self.save_o2m == 3:
                        query_o2m = query.clone() # without aux ffn
                    
                else:
                    query = self.norms[norm_index](query)
                    norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                if ffn_index == 0:
                    query_o2m = self.ffns[ffn_index](
                        query, te_query, lclc_adj[:, 1:2], lcte_adj[:, 1:2], te_cls_scores, identity=identity if self.pre_norm else None)
                else:
                    query = self.ffns[ffn_index](
                        query, te_query, lclc_adj[:, 0:1], lcte_adj[:, 0:1], te_cls_scores, identity=identity if self.pre_norm else None)
                ffn_index += 1                
                
        return query, query_o2m


@FEEDFORWARD_NETWORK.register_module()
class RATopo_FFN(BaseModule):

    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=512,
                 num_query=200,
                 num_point=11,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.1,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 edge_weight=0.5, 
                 num_te_classes=13,
                 num_lanes_one2one=300,
                 many_gcn=False,
                 detach_te_feat=False,
                 **kwargs):

        super(RATopo_FFN, self).__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_query = num_query
        self.num_point = num_point
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(
            Sequential(
                Linear(feedforward_channels, embed_dims), self.activate,
                nn.Dropout(ffn_drop)))
        self.layers = Sequential(*layers)
        self.num_lanes_one2one = num_lanes_one2one
        self.many_gcn = many_gcn
        self.edge_weight = edge_weight

        self.lclc_gnn_layer = LclcSkgGCNLayer(embed_dims, embed_dims, edge_weight=edge_weight)
        self.lcte_gnn_layer = LcteSkgGCNLayer(embed_dims, embed_dims, 
                                num_te_classes=num_te_classes, edge_weight=edge_weight, detach_te_feat=detach_te_feat)

        self.downsample = nn.Linear(embed_dims * 2, embed_dims)

        self.gnn_dropout1 = nn.Dropout(ffn_drop)
        self.gnn_dropout2 = nn.Dropout(ffn_drop)

        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self, lc_query, te_query, lclc_adj, lcte_adj, te_cls_scores, identity=None):
        # breakpoint()
        out = self.layers(lc_query)
        out = out.permute(1, 0, 2)

        '''
            out: torch.Size([b, 300 * k, 256])
            te_query: torch.Size([b, 100, 256])
            te_cls_scores: torch.Size([b, 100, 13])

            lclc_adj: torch.Size([b, k, 300, 300])
            lcte_adj: torch.Size([b, k, 300, 100])
        '''
        bs = out.size(0)
        out = out.split(self.num_lanes_one2one, dim=1) # [b, 300, 256] * k
        
        out_one = out[0] # b, 300, 256
        lclc_adj_one = lclc_adj[:, 0] # b, 300, 300
        lcte_adj_one = lcte_adj[:, 0] # b, 300, 100
        lclc_features_one = self.lclc_gnn_layer(out_one, lclc_adj_one)
        lcte_features_one = self.lcte_gnn_layer(te_query, lcte_adj_one, te_cls_scores)
        out_one = torch.cat([lclc_features_one, lcte_features_one], dim=-1)
        out_one = self.activate(out_one)
        out_one = self.gnn_dropout1(out_one)
        out_one = self.downsample(out_one)
        out_one = self.gnn_dropout2(out_one)

        if len(out) > 1:
            out_many = torch.cat(out[1:], dim=0) # b * (k - 1), 300, 256
            lclc_adj_many = lclc_adj[:, 1:].flatten(0, 1) # b * (k - 1), 300, 300
            lcte_adj_many = lcte_adj[:, 1:].flatten(0, 1) # b * (k - 1), 300, 100
            if self.many_gcn:
                lclc_features_many = self.lclc_gnn_layer(out_many, lclc_adj_many)
                lcte_features_many = self.lcte_gnn_layer(te_query.repeat(lcte_adj_many.size(0) // bs, 1, 1), 
                                                         lcte_adj_many, 
                                                         te_cls_scores.repeat(lcte_adj_many.size(0) // bs, 1, 1))
                out_many = torch.cat([lclc_features_many, lcte_features_many], dim=-1)
                out_many = self.activate(out_many)
                out_many = self.gnn_dropout1(out_many)
                out_many = self.downsample(out_many)
                out_many = self.gnn_dropout2(out_many)

            out = torch.cat([out_one, out_many], dim=0)
        else:
            out = out_one

        out = torch.cat(out.split(bs, dim=0), dim=1)
        out = out.permute(1, 0, 2)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = lc_query
        return identity + self.dropout_layer(out)

class LclcSkgGCNLayer(nn.Module):

    def __init__(self, in_features, out_features, edge_weight=0.5):
        super(LclcSkgGCNLayer, self).__init__()
        self.edge_weight = edge_weight

        if self.edge_weight != 0:
            self.weight_forward = torch.Tensor(in_features, out_features)
            self.weight_forward = nn.Parameter(nn.init.xavier_uniform_(self.weight_forward))
            self.weight_backward = torch.Tensor(in_features, out_features)
            self.weight_backward = nn.Parameter(nn.init.xavier_uniform_(self.weight_backward))

        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        self.edge_weight = edge_weight

    def forward(self, input, adj):

        support_loop = torch.matmul(input, self.weight)
        output = support_loop

        if self.edge_weight != 0:
            support_forward = torch.matmul(input, self.weight_forward)
            output_forward = torch.matmul(adj, support_forward)
            output += self.edge_weight * output_forward

            support_backward = torch.matmul(input, self.weight_backward)
            output_backward = torch.matmul(adj.permute(0, 2, 1), support_backward)
            output += self.edge_weight * output_backward

        return output

class LclcSkgGCNLayerOneway(nn.Module):

    def __init__(self, in_features, out_features, edge_weight=0.5):
        super(LclcSkgGCNLayerOneway, self).__init__()    
        self.weight_forward = torch.Tensor(in_features, out_features)
        self.weight_forward = nn.Parameter(nn.init.xavier_uniform_(self.weight_forward))
        self.edge_weight = edge_weight

    def forward(self, input, adj):
        support_forward = torch.matmul(input, self.weight_forward)
        output_forward = torch.matmul(adj, support_forward)
        output = self.edge_weight * output_forward
        return output

class LcteSkgGCNLayer(nn.Module):

    def __init__(self, in_features, out_features, num_te_classes=13, edge_weight=0.5, detach_te_feat=False):
        super(LcteSkgGCNLayer, self).__init__()
        self.weight = torch.Tensor(num_te_classes, in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        self.edge_weight = edge_weight
        self.detach_te_feat = detach_te_feat

    def forward(self, te_query, lcte_adj, te_cls_scores):
        # te_cls_scores: (bs, num_te_query, num_te_classes)
        cls_scores = te_cls_scores.detach().sigmoid().unsqueeze(3)
        # te_query: (bs, num_te_query, embed_dims)
        # (bs, num_te_query, 1, embed_dims) * (bs, num_te_query, num_te_classes, 1)
        te_feats = te_query.unsqueeze(2) * cls_scores
        if self.detach_te_feat:
            te_feats = te_feats.clone().detach()
        # (bs, num_te_classes, num_te_query, embed_dims)
        te_feats = te_feats.permute(0, 2, 1, 3)

        support = torch.matmul(te_feats, self.weight).sum(1)
        adj = lcte_adj * self.edge_weight
        output = torch.matmul(adj, support)
        return output

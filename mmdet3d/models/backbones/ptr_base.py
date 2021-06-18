import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmdet3d.ops import (PointFPModule, Points_Sampler, QueryAndGroup,
                         gather_points)
from mmdet.models import BACKBONES

class TransformerEncoderLayerPreNorm(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):

        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)

        self.activation = nn.ReLU(inplace=True)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask = None, src_key_padding_mask = None):

        src = self.norm1(src)
        src2, mask = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        
        return src

class TransformerDecoderLayerPreNorm(nn.Module):

    def __init__(self, d_model, nc_mem, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm_mem = nn.LayerNorm(nc_mem)

        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.dropout3 = nn.Dropout(dropout, inplace=True)

        self.activation = nn.ReLU(inplace=True)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        tgt = self.norm1(tgt)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt = self.norm2(tgt)
        memory = self.norm_mem(memory)
        tgt2, mask = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt

class LinformerEncoderLayer(nn.Module):

    def __init__(self, src_len, ratio, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):

        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear_k = nn.Parameter(torch.empty(src_len // ratio, src_len))
        self.linear_v = nn.Parameter(torch.empty(src_len // ratio, src_len))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)

        self.activation = nn.ReLU(inplace=True)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_k)
        nn.init.xavier_uniform_(self.linear_v)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

class LinformerDecoderLayer(nn.Module):

    def __init__(self, tgt_len, mem_len, ratio, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.dropout3 = nn.Dropout(dropout, inplace=True)

        self.linear_k1 = nn.Parameter(torch.empty(tgt_len // ratio, tgt_len))
        self.linear_v1 = nn.Parameter(torch.empty(tgt_len // ratio, tgt_len))
        self.linear_k2 = nn.Parameter(torch.empty(mem_len // ratio, mem_len))
        self.linear_v2 = nn.Parameter(torch.empty(mem_len // ratio, mem_len))

        self.activation = nn.ReLU(inplace=True)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_k1)
        nn.init.xavier_uniform_(self.linear_v1)
        nn.init.xavier_uniform_(self.linear_k2)
        nn.init.xavier_uniform_(self.linear_v2)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        tgt_temp = tgt.transpose(0, 1)
        key = torch.matmul(self.linear_k1, tgt_temp).transpose(0, 1)
        value = torch.matmul(self.linear_v1, tgt_temp).transpose(0, 1)
        tgt2 = self.self_attn(tgt, key, value, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        memory_temp = memory.transpose(0, 1)
        key = torch.matmul(self.linear_k2, memory_temp).transpose(0, 1)
        value = torch.matmul(self.linear_v2, memory_temp).transpose(0, 1)
        tgt2 = self.multihead_attn(tgt, key, value, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class LocalTransformer(nn.Module):

    def __init__(self, npoint, radius, nsample, dim_feature, dim_out, nhead=4, num_layers=2, norm_cfg=dict(type='BN2d'), ratio=1, drop=0.0, prenorm=True):
        super().__init__()

        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        self.nc_in = dim_feature
        self.nc_out = dim_out
        
        self.sampler = Points_Sampler([self.npoint], ['D-FPS'])
        self.grouper = QueryAndGroup(self.radius, self.nsample, use_xyz=False, return_grouped_xyz=True, normalize_xyz=False)

        self.pe = nn.Sequential(
            ConvModule(3, self.nc_in // 2, 1, norm_cfg=norm_cfg),
            ConvModule(self.nc_in // 2, self.nc_in, 1, act_cfg=None, norm_cfg=None)
            )

        BSC_Encoder = TransformerEncoderLayerPreNorm if prenorm else nn.TransformerEncoderLayer
        
        self.chunk = nn.TransformerEncoder(
            BSC_Encoder(d_model=self.nc_in, dim_feedforward=2 * self.nc_in, dropout=drop, nhead=nhead) if ratio == 1 else
            LinformerEncoderLayer(src_len=nsample, ratio=ratio, d_model=self.nc_in, nhead=nhead, dropout=drop, dim_feedforward=2 * self.nc_in),
            num_layers=num_layers)
        
        self.fc = ConvModule(self.nc_in, self.nc_out, 1, norm_cfg=None, act_cfg=None)

    def forward(self, xyz, features):

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        fps_idx = self.sampler(xyz, features)
        new_xyz = gather_points(xyz_flipped, fps_idx).transpose(1, 2)
        group_features, group_xyz = self.grouper(xyz.contiguous(), new_xyz.contiguous(), features.contiguous()) # (B, 3, npoint, nsample) (B, C, npoint, nsample)
        B = group_xyz.shape[0]
        position_encoding = self.pe(group_xyz)
        input_features = group_features + position_encoding
        B, D, np, ns = input_features.shape
        
        input_features = input_features.permute(0, 2, 1, 3).reshape(-1, D, ns).permute(2, 0, 1)
        transformed_feats = self.chunk(input_features).permute(1, 2, 0).reshape(B, np, D, ns).transpose(1, 2)
        output_features = F.max_pool2d(transformed_feats, kernel_size=[1, ns])  # (B, C, npoint)
        output_features = self.fc(output_features).squeeze(-1)

        return new_xyz, output_features, fps_idx
        

class GlobalTransformer(nn.Module):

    def __init__(self, dim_feature, dim_out, nhead=4, num_layers=2, norm_cfg=dict(type='BN2d'), ratio=1, src_pts=2048, drop=0.0, prenorm=True):
        
        super().__init__()

        self.nc_in = dim_feature
        self.nc_out = dim_out
        self.nhead = nhead
        
        self.pe = nn.Sequential(
            ConvModule(3, self.nc_in // 2, 1, norm_cfg=norm_cfg),
            ConvModule(self.nc_in // 2, self.nc_in, 1, act_cfg=None, norm_cfg=None)
            )
        
        BSC_Encoder = TransformerEncoderLayerPreNorm if prenorm else nn.TransformerEncoderLayer

        self.chunk = nn.TransformerEncoder(
            BSC_Encoder(d_model=self.nc_in, dim_feedforward=2 * self.nc_in, dropout=drop, nhead=nhead) if ratio == 1 else
            LinformerEncoderLayer(src_len=src_pts, ratio=ratio, d_model=self.nc_in, nhead=nhead, dropout=drop, dim_feedforward=2 * self.nc_in), 
            num_layers=num_layers)
        
        self.fc = ConvModule(self.nc_in, self.nc_out, 1, norm_cfg=None, act_cfg=None)

    def forward(self, xyz, features):

        xyz_flipped = xyz.transpose(1, 2).unsqueeze(-1)
        input_features = features.unsqueeze(-1) + self.pe(xyz_flipped)
        input_features = input_features.squeeze(-1).permute(2, 0, 1)
        transformed_feats = self.chunk(input_features).permute(1, 2, 0)
        output_features = self.fc(transformed_feats.unsqueeze(-1)).squeeze(-1)
        
        return output_features

class LocalGlobalTransformer(nn.Module):

    def __init__(self, dim_in, dim_out, nhead=4, num_layers=2, norm_cfg=dict(type='BN2d'), ratio=1, mem_pts=20000, tgt_pts=2048, drop=0.0, dim_feature=64, prenorm=True):

        super().__init__()
        
        self.nc_in = dim_in
        self.nc_out = dim_out
        self.nhead = nhead
        self.pe = nn.Sequential(
            ConvModule(3, self.nc_in // 2, 1, norm_cfg=norm_cfg),
            ConvModule(self.nc_in // 2, self.nc_in, 1, act_cfg=None, norm_cfg=None)
            )

        BSC_Decoder = TransformerDecoderLayerPreNorm if prenorm else nn.TransformerDecoderLayer
        
        self.chunk = nn.TransformerDecoder(
            BSC_Decoder(d_model=self.nc_in, dim_feedforward=2 * self.nc_in, dropout=drop, nhead=nhead, nc_mem=dim_feature) if ratio == 1 else
            LinformerDecoderLayer(tgt_len=tgt_pts, mem_len=mem_pts, ratio=ratio, d_model=self.nc_in, nhead=nhead, dropout=drop, dim_feedforward=2 * self.nc_in),
            num_layers=num_layers)
        
        self.fc = ConvModule(self.nc_in, self.nc_out, 1, norm_cfg=None, act_cfg=None)

    def forward(self, xyz_tgt, xyz_mem, features_tgt, features_mem):

        xyz_tgt_flipped = xyz_tgt.transpose(1, 2).unsqueeze(-1)
        xyz_mem_flipped = xyz_mem.transpose(1, 2).unsqueeze(-1)
        
        tgt = features_tgt.unsqueeze(-1) + self.pe(xyz_tgt_flipped)
        mem = features_mem.unsqueeze(-1) + self.pe(xyz_mem_flipped)

        mem_mask = None
        
        mem = mem.squeeze(-1).permute(2, 0, 1)
        tgt = tgt.squeeze(-1).permute(2, 0, 1)

        transformed_feats = self.chunk(tgt, mem, memory_mask=mem_mask).permute(1, 2, 0)
        output_features = self.fc(transformed_feats.unsqueeze(-1)).squeeze(-1)
        
        return output_features

class BasicDownBlock(nn.Module):
    
    def __init__(self, 
                 npoint, radius, nsample, 
                 dim_feature, dim_hid, dim_out, 
                 nhead=4, num_layers=2, 
                 norm_cfg=dict(type='BN2d'), 
                 ratio=1, mem_pts=20000, use_lin_enc=False,
                 use_decoder=True,
                 local_drop=0.0, global_drop=0.0, decoder_drop=0.0,
                 prenorm=True):
        
        super().__init__()
        self.use_decoder = use_decoder
        enc_ratio = ratio if use_lin_enc else 1
        self.local_chunk = LocalTransformer(npoint, radius, nsample, dim_feature, dim_hid, nhead, num_layers, norm_cfg, enc_ratio, local_drop, prenorm)
        self.global_chunk = GlobalTransformer(dim_hid, dim_out, nhead, num_layers, norm_cfg, enc_ratio, npoint, global_drop, prenorm)
        if use_decoder:
            self.combine_chunk = LocalGlobalTransformer(dim_hid, dim_hid, nhead, 1, norm_cfg, ratio, mem_pts, npoint, decoder_drop, dim_feature, prenorm)

    def forward(self, xyz, features):

        new_xyz, local_features, fps_idx = self.local_chunk(xyz, features)

        if self.use_decoder:
            combined = self.combine_chunk(new_xyz, xyz, local_features, features)
            combined += local_features
        else:
            combined = local_features
        output_feats = self.global_chunk(new_xyz, combined)

        return new_xyz, output_feats, fps_idx

@BACKBONES.register_module()
class Pointformer(nn.Module):

    def init_weights(self, pretrained=None):
        pass

    def __init__(self, 
                 num_points=(2048, 1024, 512, 256),
                 radius=(0.2, 0.4, 0.8, 1.2),
                 num_samples=(16, 16, 16, 16),
                 basic_channels=64,
                 fp_channels=((256, 256), (256, 256)),
                 num_heads=4,
                 num_layers=2,
                 ratios=(1, 1, 1, 1),
                 use_decoder=(True, True, True, True),
                 use_lin_enc=False,
                 cloud_points=20000,
                 local_drop=0.0, global_drop=0.0, decoder_drop=0.0,
                 norm_cfg=dict(type='BN2d'),
                 prenorm=True
                 ):

        super().__init__()
        bc = basic_channels
        self.num_sa = len(num_points)
        self.SA_modules = nn.ModuleList()
        self.nc_outs = []

        self.feature_conv = ConvModule(1, bc, 1, norm_cfg=None, act_cfg=None)
        
        mem_pts = cloud_points

        for i, (np, r, ns, ratio, use) in enumerate(zip(num_points, radius, num_samples, ratios, use_decoder)):
            if i < self.num_sa - 1:
                self.SA_modules.append(BasicDownBlock(np, r, ns, bc, bc, bc * 2, num_heads, num_layers, norm_cfg, ratio, mem_pts, use_lin_enc, use, local_drop, global_drop, decoder_drop, prenorm))
                self.nc_outs.append(bc * 2)
            else:
                self.SA_modules.append(BasicDownBlock(np, r, ns, bc, bc, bc, num_heads, num_layers, norm_cfg, ratio, mem_pts, use_lin_enc, use, local_drop, global_drop, decoder_drop, prenorm))
                self.nc_outs.append(bc)
            
            bc = bc * 2
            mem_pts = np

        self.FP_modules = nn.ModuleList(
            [
                PointFPModule([self.nc_outs[-1] + self.nc_outs[-2], fp_channels[0][0], fp_channels[0][1]], norm_cfg=norm_cfg),
                PointFPModule([self.nc_outs[-3] + fp_channels[0][1], fp_channels[1][0], fp_channels[1][1]], norm_cfg=norm_cfg)
            ]
        )
        self.num_fp = len(self.FP_modules)

    def forward(self, points):
        
        xyz = points[...,0:3].contiguous()
        features = self.feature_conv(points[...,3:].transpose(1, 2).unsqueeze(-1)).squeeze(-1)
        batch, num_points = xyz.shape[:2]
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(batch, 1).long()

        sa_xyz = [xyz]
        sa_features = [features]
        sa_indices = [indices]

        for i in range(self.num_sa):
            cur_xyz, cur_features, cur_indices = self.SA_modules[i](
                sa_xyz[i], sa_features[i])
            sa_xyz.append(cur_xyz.contiguous())
            sa_features.append(cur_features.contiguous())
            sa_indices.append(
                torch.gather(sa_indices[-1], 1, cur_indices.long()))

        fp_xyz = [sa_xyz[-1]]
        fp_features = [sa_features[-1]]
        fp_indices = [sa_indices[-1]]
        for i in range(self.num_fp):
            fp_features.append(self.FP_modules[i](
                sa_xyz[self.num_sa - i - 1], sa_xyz[self.num_sa - i],
                sa_features[self.num_sa - i - 1], fp_features[-1]))
            fp_xyz.append(sa_xyz[self.num_sa - i - 1])
            fp_indices.append(sa_indices[self.num_sa - i - 1])

        ret = dict(fp_xyz=fp_xyz, fp_features=fp_features, fp_indices=fp_indices)
        
        return ret

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from transformers import AutoFeatureExtractor, ViTMAEModel
from torch import Tensor
from torch.nn.modules.loss import _Loss
from mca import *
  


class Highway(nn.Module):
    r"""Highway Layers
    Args:
        - num_highway_layers(int): number of highway layers.
        - input_size(int): size of highway input.
    """

    def __init__(self, num_highway_layers, input_size):
        super(Highway, self).__init__()
        self.num_highway_layers = num_highway_layers
        self.non_linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        for layer in range(self.num_highway_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            # Compute percentage of non linear information to be allowed for each element in x
            non_linear = F.relu(self.non_linear[layer](x))
            # Compute non linear information
            linear = self.linear[layer](x)
            # Compute linear information
            x = gate * non_linear + (1 - gate) * linear
            # Combine non linear and linear information according to gate
            # x = self.dropout(x)
        return x  


class SELU(nn.Module):

    def __init__(self, alpha=1.6732632423543772848170429916717,
                 scale=1.0507009873554804934193349852946, inplace=False):
        super(SELU, self).__init__()

        self.scale = scale
        self.elu = nn.ELU(alpha=alpha, inplace=inplace)

    def forward(self, x):
        return self.scale * self.elu(x)


class NeimsBlock(nn.Module):
    """
    Adapted from: https://github.com/brain-research/deep-molecular-massspec
    Copyright (c) 2018 Google LLC
    See licenses/NEIMS_LICENSE
    """

    def __init__(self,in_dim,out_dim,dropout=0.5):
        
        super(NeimsBlock, self).__init__()
        bottleneck_factor = 0.5
        bottleneck_size = int(round(bottleneck_factor*out_dim)) if in_dim < 2048 else 256
        bottleneck_size = 256
        self.in_batch_norm = nn.LayerNorm(bottleneck_size)
        self.in_activation =   nn.ReLU()
        self.in_linear = nn.Linear(in_dim, bottleneck_size)
        # self.out_batch_norm = nn.LayerNorm(out_dim)
        self.out_linear = nn.Linear(bottleneck_size, out_dim)
        self.out_activation =   nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.attention_gated_part1_1 = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=bottleneck_size),
             
            nn.Tanh(),
        )
        self.attention_gated_part1_2 = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=bottleneck_size),
             
            nn.Sigmoid(),
        )
        self.attention_gated_part2 = nn.Linear(in_features=bottleneck_size, out_features=out_dim) 
        self.dropout = nn.Dropout(0.2)
       
    def forward(self,x):
        h = x
        h = self.in_linear(h)
        h = self.in_activation(h)
        # h = self.in_batch_norm(h)
        # h = self.dropout(h)
        h = self.out_linear(h)
        h = self.out_activation(h)
        h = self.dropout(h)
        # h = self.out_batch_norm(h)
        # V = self.attention_gated_part1_1(h)    # N * D
        # U = self.attention_gated_part1_2(h)    # N * D
        # h = self.attention_gated_part2(  V * U)
        # h = self.dropout(h)
        return  h


 

class Fusion_modal(nn.Module):
    def __init__(self, embed_dim, dynamic_fusion_dim = 128):
        super(Fusion_modal, self).__init__()
        n_heads = 8
        opt = {"fusion":{
        'mca_HIDDEN_SIZE': embed_dim,
        'mca_DROPOUT_R': 0.1,
        'mca_FF_SIZE': 1024,
        'mca_MULTI_HEAD': n_heads,
        'mca_HIDDEN_SIZE_HEAD': embed_dim//n_heads,
        }}

        # local trans
        self.l2l_SA_1 = SA(opt)
        self.l2l_SA_2 = SA(opt)

        # cross trans
        self.cross_SGA_1 = SGA(opt)
        self.cross_SGA_2 = SGA(opt)

        self.dropout = nn.Dropout(0.5)

        # self.proj = nn.Linear(embed_dim, embed_dim) 
        # # dynamic fusion
        # self.dynamic_weight = nn.Sequential(
        #     nn.Linear(embed_dim, dynamic_fusion_dim),
        #     nn.Sigmoid(),
        #     nn.Dropout(0.1),
        #     nn.Linear(dynamic_fusion_dim, 2),
        #     nn.Softmax()
        # )

        # # self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_feature, fp_feature):
        
        img_feature = torch.unsqueeze(img_feature, dim=1)
        fp_feature = torch.unsqueeze(fp_feature, dim=1)

       
        # img_feature = self.l2l_SA_1(img_feature)
        # fp_feature = self.l2l_SA_2(fp_feature)

        img_feature = self.cross_SGA_1(img_feature, fp_feature)
        fp_feature = self.cross_SGA_2(fp_feature, img_feature)

        img_feature = torch.squeeze(img_feature, dim=1)
        fp_feature = torch.squeeze(fp_feature, dim=1)
        feature_if =  img_feature + fp_feature
        # feature_if = self.proj(torch.cat([img_feature, fp_feature], axis=1))
        # dynamic fusion
        # feature_if = img_feature    + fp_feature
        # feature_if = self.dropout(feature_if)
        # feature_if = self.proj(feature_if)
        # dynamic_weight = self.dynamic_weight(feature_if)

        # weight_img = dynamic_weight[:, 0].reshape(feature_if.shape[0],-1).expand_as(img_feature)
        # weight_fp = dynamic_weight[:, 1].reshape(feature_if.shape[0],-1).expand_as(fp_feature)

        # fusion_feature = weight_img*img_feature + weight_fp*fp_feature
        # fusion_feature = self.norm(fusion_feature)
        # feature_if = self.highway(feature_if)
        return   feature_if

class HiFusion_model(nn.Module):
    def __init__(self, embed_dim, dynamic_fusion_dim = 256):
        super(HiFusion_model, self).__init__()
        
        self.fusion_fp1cell = Fusion_modal(embed_dim, dynamic_fusion_dim)
        self.fusion_fp2cell = Fusion_modal(embed_dim,dynamic_fusion_dim)
        self.fusion_final1 = Fusion_modal(embed_dim, dynamic_fusion_dim)
        self.fusion_final2 = Fusion_modal(embed_dim, dynamic_fusion_dim)
        self.fusion_final3 = Fusion_modal(embed_dim, dynamic_fusion_dim)

        self.fusion_fpfp = Fusion_modal(embed_dim, dynamic_fusion_dim)
        self.highway = Highway(2, embed_dim)
        

        
    def forward(self, cell_feature, fp1_feature, fp2_feature):

        fp1_cell = self.fusion_fp1cell(cell_feature, fp1_feature)
        fp2_cell = self.fusion_fp2cell(cell_feature, fp2_feature)
        fusion_feature = self.fusion_final1(fp1_cell, fp2_feature)

        fpfp = self.fusion_fpfp(fp1_feature, fp2_feature)
        fusion_feature2 = self.fusion_final2(fpfp, cell_feature)
        # fusion_feature = torch.cat([fusion_feature, fusion_feature2 ], 1)
        fusion_feature = self.fusion_final3(fusion_feature, fusion_feature2)
        return fusion_feature
 

class MAG(nn.Module):
    def __init__(self, cell_dim, beta_shift=1, dropout_prob=0.5):
        super(MAG, self).__init__()
         
        self.W_drugcell1 = nn.Linear(2*cell_dim, cell_dim)
        self.W_drugcell2 = nn.Linear(2*cell_dim, cell_dim)

        self.W_drug2cell1 = nn.Linear(cell_dim, cell_dim)
        self.W_drug2cell2 = nn.Linear(cell_dim, cell_dim)

        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(cell_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, cell_features, drug1_feature, drug2_feature):
        eps = 1e-6
         
        # import ipdb;ipdb.set_trace()
        weight_dc1 = F.relu(self.W_drugcell1(torch.cat((drug1_feature, cell_features), dim=-1)))
        weight_dc2 = F.relu(self.W_drugcell2(torch.cat((drug2_feature, cell_features), dim=-1)))

        h_m = weight_dc1 * self.W_drug2cell1(drug1_feature) + weight_dc2 * self.W_drug2cell2(drug2_feature)

        em_norm = cell_features.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)
        DEVICE = cell_features.device

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(DEVICE)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(DEVICE)

        alpha = torch.min(thresh_hold, ones)

        alpha = alpha.unsqueeze(dim=-1)

        drug_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(drug_embedding + cell_features)
        )
        # embedding_output = drug_embedding + cell_features

        return embedding_output 

class FPDDS(nn.Module):
    def __init__(self,
                 cell_dim, descriptor_size):
        super(FPDDS, self).__init__()
        
         
        cell_output_dim = 512
       
        
        self.cell_line_mlp =  NeimsBlock(cell_dim, cell_output_dim)  
        self.fp_mlp =  NeimsBlock(1024, cell_output_dim)  
        self.descript_mlp = NeimsBlock(descriptor_size, cell_output_dim)  
 

        self.predict = nn.Sequential(
                Highway(2, cell_output_dim),
                nn.Dropout(0.5),
                nn.Linear( cell_output_dim, 1),  
                nn.Sigmoid()
                )

         

        self.fusion_model = MAG(cell_output_dim)
        self.fusion_drug1 = Fusion_modal(cell_output_dim)
        self.fusion_drug2 = Fusion_modal(cell_output_dim)
        # self.fusion_model = HiFusion_model(cell_output_dim)
         
 

    def forward(self,  fp11, fp12, descript1, descript2,   cell_features):
          
        
        cell_features = self.cell_line_mlp(cell_features)
        fp11_feature = self.fp_mlp(fp11)
        fp12_feature = self.fp_mlp(fp12)

        descript1_feature = self.descript_mlp(descript1)
        descript2_feature = self.descript_mlp(descript2)

        drug1_features = self.fusion_drug1(descript1_feature, fp11_feature)
        drug2_features = self.fusion_drug1(descript2_feature, fp12_feature)

       

        final_features = self.fusion_model(cell_features, drug1_features, drug2_features )  
        # final_features2 = self.fusion_model(cell_features, fp21_feature, fp22_feature)
        # final_features2 = torch.cat([cell_features, drug1_features, drug2_features], 1)
        predict = self.predict(final_features)

        
        
        return predict 


class ProjBlock(nn.Module):
    def __init__(self,in_dim,out_dim,dropout=0.5):
        
        super(ProjBlock, self).__init__()
        # bottleneck_factor = 0.5
        # bottleneck_size = int(round(bottleneck_factor*out_dim)) if in_dim < 2048 else 256
        bottleneck_size = 256
         

        self.attention_gated_part1_1 = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=bottleneck_size),
            nn.Tanh(),
        )
        self.attention_gated_part1_2 = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=bottleneck_size),
            nn.Sigmoid(),
        )
        self.attention_gated_part2 = nn.Linear(in_features=bottleneck_size, out_features=out_dim) 
        self.dropout = nn.Dropout(0.2)
       
    def forward(self,x):
        h = x
        V = self.attention_gated_part1_1(h)    # N * D
        U = self.attention_gated_part1_2(h)    # N * D
        h = self.attention_gated_part2(  V * U)
        h = self.dropout(h)
        return  h




class ResidualFUsionBlock(nn.Module):
    def __init__(self,in_dim,dropout=0.5):
        
        super(ResidualFUsionBlock, self).__init__()
        
 
        self.attention_gated_part_1 = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=in_dim),
            nn.Sigmoid(),
        )

        self.attention_gated_part_2 = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=in_dim),
            nn.Sigmoid(),
        )
        
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_dim*2, out_features=in_dim),
            nn.ReLU(),
        )
    def forward(self,x1, x2):

        
        att_x1 = self.attention_gated_part_1(x1)    
        att_x2 = self.attention_gated_part_2(x2)     
        
        x1 = x1 + att_x2 * x1 
        x2 = x2 + att_x1 * x2 
        x = self.fc( torch.cat( (x1, x2), dim=1 ))
        return  x 

class ImageDDS(nn.Module):
    def __init__(self,
                 model_filename,
                 cell_dim):
        super(ImageDDS, self).__init__()
        self.drug_encoder = ViTMAEModel.from_pretrained(model_filename, output_hidden_states=True) 
        
        for param in self.drug_encoder.parameters():
            param.requires_grad = False

        cell_output_dim = 256
        hidden_dim = 1024

         
        self.cell_line_mlp = ProjBlock(cell_dim, cell_output_dim, 0.5)
        self.d1_mlp = ProjBlock(768, cell_output_dim, 0.5)
 

        # self.fusion = latefusion(cell_output_dim)
        # self.fusion = Highway(2, cell_output_dim*3)
        self.fusion_d1c = ResidualFUsionBlock(cell_output_dim)
        self.fusion_d2c = ResidualFUsionBlock(cell_output_dim)
        self.fusion_d1d2 = ResidualFUsionBlock(cell_output_dim)
        
        self.predict = nn.Sequential(
                nn.Linear( cell_output_dim*3, 1),
                nn.Sigmoid()
                )

        
 
 

    def forward(self, drug1_inputs, drug2_inputs,fp1, fp2, cell_features):
          
        drug1_feature = self.drug_encoder(drug1_inputs)
        drug1_feature = drug1_feature.last_hidden_state[:, 0, :]
        drug2_feature = self.drug_encoder(drug2_inputs)
        drug2_feature = drug2_feature.last_hidden_state[:, 0, :]


        cell_features = self.cell_line_mlp(cell_features)
        drug1_feature = self.d1_mlp(drug1_feature)
        drug2_feature = self.d1_mlp(drug2_feature)

        d1c = self.fusion_d1c(drug1_feature, cell_features)
        d2c = self.fusion_d1c(drug2_feature, cell_features)
        d1d2 = self.fusion_d1d2(drug2_feature, drug1_feature)
        
        final_features = torch.cat([d1c, d2c, d1d2], axis=1)
        predict = self.predict(final_features)
        
        return predict
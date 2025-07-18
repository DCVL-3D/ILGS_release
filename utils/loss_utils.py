# Copyright (C) 2025, DCVL-3D
# ILGS_release research group, https://github.com/DCVL-3D/ILGS_release
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from scipy.spatial import cKDTree
import math

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def masked_l1_loss(network_output, gt, mask):
    mask = mask.float()[None,:,:].repeat(gt.shape[0],1,1)
    loss = torch.abs((network_output - gt)) * mask
    loss = loss.sum() / mask.sum()
    return loss

def weighted_l1_loss(network_output, gt, weight):
    loss = torch.abs((network_output - gt)) * weight
    return loss.mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

    
def loss_semantic_3d(features, ids, clip_features, k=10, lambda_val=2.0, 
                                      max_points=200000, sample_size=1000, margin=0.5, random_neighbor_count=50):

    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0))[:max_points]
        features = features[indices]
        ids = ids[indices]
        clip_features = clip_features[indices]
    
    # sample_size 개수만큼 무작위 샘플 선택
    indices = torch.randperm(features.size(0))[:sample_size]
    sample_features = features[indices]     
    sample_ids = ids[indices]                
    sample_clips = clip_features[indices]    
    
    half = sample_size // 2  
    
    # knn
    sample_features_knn = sample_features[:half]   
    sample_ids_knn = sample_ids[:half]             
    sample_clips_knn = sample_clips[:half]         

    dists_A = torch.cdist(sample_features_knn, features)  
    _, knn_indices = dists_A.topk(k, largest=False)        
    knn_neighbor_ids = ids[knn_indices]                   
    knn_neighbor_clips = clip_features[knn_indices]      
    
    # 그룹 B: 랜덤 neighbor 방식
    sample_features_rand = sample_features[half:]    
    sample_ids_rand = sample_ids[half:]               
    sample_clips_rand = sample_clips[half:]          
    
    group_B_size = sample_features_rand.size(0)       
    rand_indices = torch.randint(0, features.size(0), (group_B_size, random_neighbor_count), device=features.device)
    rand_neighbor_ids = ids[rand_indices]            
    rand_neighbor_clips = clip_features[rand_indices] 
    
  
    sample_clips_knn_exp = sample_clips_knn.unsqueeze(1).expand(-1, k, -1)  
    cos_sim_knn = F.cosine_similarity(sample_clips_knn_exp, knn_neighbor_clips, dim=-1) 
    

    same_mask_knn = (knn_neighbor_ids == sample_ids_knn.unsqueeze(1)) 
    diff_mask_knn = ~same_mask_knn
    
   
    sim_loss_knn = (1 - cos_sim_knn) * same_mask_knn.float()
    diff_loss_knn = torch.clamp(1 + cos_sim_knn - margin, min=0) * diff_mask_knn.float()
    total_same_knn = same_mask_knn.float().sum()
    total_diff_knn = diff_mask_knn.float().sum()
    

    sample_clips_rand_exp = sample_clips_rand.unsqueeze(1).expand(-1, random_neighbor_count, -1)  
    cos_sim_rand = F.cosine_similarity(sample_clips_rand_exp, rand_neighbor_clips, dim=-1)  
    

    same_mask_rand = (rand_neighbor_ids == sample_ids_rand.unsqueeze(1))  
    diff_mask_rand = ~same_mask_rand
    
 
    sim_loss_rand = (1 - cos_sim_rand) * same_mask_rand.float()
    diff_loss_rand = torch.clamp(1 + cos_sim_rand - margin, min=0) * diff_mask_rand.float()
    total_same_rand = same_mask_rand.float().sum()
    total_diff_rand = diff_mask_rand.float().sum()
    
 
    total_sim_loss = sim_loss_knn.sum() + sim_loss_rand.sum()
    total_diff_loss = diff_loss_knn.sum() + diff_loss_rand.sum()
    total_same = total_same_knn + total_same_rand
    total_diff = total_diff_knn + total_diff_rand
    

    sim_loss = total_sim_loss / total_same if total_same > 0 else 0.0
    diff_loss = total_diff_loss / total_diff if total_diff > 0 else 0.0
    
    total_loss = sim_loss + diff_loss
    return lambda_val * total_loss

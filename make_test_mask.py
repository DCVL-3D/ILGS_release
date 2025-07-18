#############################################################
# LeRF
#############################################################
# import torch
# import numpy as np
# import os
# from PIL import Image
# import open_clip as clip

# # Set device to GPU or CPU
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load CLIP model
# model, _, _ = clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k", precision="fp32")
# model.eval()
# model = model.to(device)

# # Text query list
# text_queries = [
#     #  "baseball", "dinosaur", "rabbit", "shrilling_chicken", "weaving basket", "wood wall"
#     # "banana","black leather shoe", "camera", "hand","red bag", "white sheet"
#     # "dressing_doll","green_grape","mini_offroad_car","orange_cat","pebbled_concrete_wall","Portuguese_egg_target","wood"
#     # "black_headphone","green_lawn","hand_soap","New_York_Yankees_cap","red_apple","stapler"
#     # "a red Nintendo Switch joy-con controller","a stack of UNO cards", " grey sofa", "Gundam","Pikachu","Xbox wireless controller"
#     # "chopsticks", "egg", "glass of water", "pork belly", "wavy noodles in bowl", "yellow bowl"
#     # "green apple", "green toy chair", "old camera", "porcelain hand", "red apple", "red toy chair", "rubber duck with red hat"
#     "apple", "bag of cookies","coffee mug", "cookies on the plate", "napkin", "plate", "sheep","spoon handle", "stuffed bear","tea in a glass"
    
# ]

# # Data paths
# decoded_path = "/mnt/jsm/gaussian-grouping/output/lerf/teatime/test/ours_30000_text/feature_map_npy/decoded/"
# logits_path = "/mnt/jsm/gaussian-grouping/output/lerf/teatime/test/ours_30000_text/logits/"
# save_base_path = "/mnt/jsm/gaussian-grouping/output/lerf/teatime/test/ours_30000_text/test_mask/"

# # Define 4-connected neighborhood for finding neighbors
# # neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# # 거리 d 이내의 모든 이웃 계산
# d = 2  # 원하는 거리
# neighbor_offsets = [(dy, dx) for dy in range(-d, d + 1) for dx in range(-d, d + 1) if not (dy == 0 and dx == 0)]


# # Generate masks for each file and text query
# for idx in range(6):
#     # Load feature map
#     decoded_feature_map = np.load(f"{decoded_path}decoded_{idx:05d}.npy")  # Shape: (H, W, C)
#     logits = np.load(f"{logits_path}{idx:05d}_logits.npy")  # Shape: (H, W)
#     H, W, C = decoded_feature_map.shape

#     # Convert feature map and logits to tensors
#     decoded_feature_tensor = torch.from_numpy(decoded_feature_map).float().to(device)
#     logits_tensor = torch.from_numpy(logits).long().to(device)

#     # Normalize feature tensor
#     decoded_feature_flattened = decoded_feature_tensor.view(-1, C)
#     decoded_feature_flattened = decoded_feature_flattened / decoded_feature_flattened.norm(dim=-1, keepdim=True)

#     # Set save path
#     save_path = os.path.join(save_base_path, str(idx))
#     os.makedirs(save_path, exist_ok=True)

#     # Process each text query
#     for input_text in text_queries:
#         # Encode text query
#         text_tokens = clip.tokenize([input_text]).to(device)
#         with torch.no_grad():
#             text_feature = model.encode_text(text_tokens).float()
#         text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

#         # Cosine similarity calculation
#         cosine_similarity = torch.matmul(decoded_feature_flattened, text_feature.T).squeeze()
#         cosine_similarity = cosine_similarity.view(H, W)

#         # Calculate average similarity for each ID
#         id_values = torch.unique(logits_tensor)
#         id_avg_similarity = {}
#         for id_value in id_values:
#             id_mask = (logits_tensor == id_value)
#             if id_mask.sum() == 0:
#                 continue
#             similarities = cosine_similarity.masked_select(id_mask)
#             if similarities.numel() > 0:
#                 # Select top 80% similarities
#                 top_80_similarities = torch.topk(similarities, int(0.8 * similarities.numel())).values
#                 avg_similarity = top_80_similarities.mean() if top_80_similarities.numel() > 0 else 0
#                 id_avg_similarity[id_value.item()] = avg_similarity

#         # Get the ID with the highest average similarity
#         sorted_ids = sorted(id_avg_similarity.items(), key=lambda x: x[1], reverse=True)
#         best_id = sorted_ids[0][0]
#         best_id_avg_similarity = id_avg_similarity[best_id]

#         # Generate a neighborhood mask for best ID
#         best_id_mask = (logits_tensor == best_id)
#         best_id_mask_np = best_id_mask.cpu().numpy()
#         neighborhood_mask = np.zeros_like(best_id_mask_np, dtype=bool)
#         for dy, dx in neighbor_offsets:
#             shifted_best_mask = np.roll(best_id_mask_np, shift=(dy, dx), axis=(0, 1))
#             neighborhood_mask |= shifted_best_mask

#         # Mask generation based on similarity comparison with best ID and neighborhood
#         mask = torch.zeros_like(logits_tensor, dtype=torch.uint8)
#         for id_value in id_values:
#             id_mask = (logits_tensor == id_value)
#             if id_mask.sum() == 0 or id_value == best_id:
#                 continue

#             # Calculate average similarity for the current ID
#             similarities = cosine_similarity.masked_select(id_mask)
#             if similarities.numel() > 0:
#                 # Select top 80% similarities
#                 top_80_similarities = torch.topk(similarities, int( 1 * similarities.numel())).values
#                 avg_similarity = top_80_similarities.mean() if top_80_similarities.numel() > 0 else 0

#                 # Compare with the best ID's average similarity
#                 similarity_difference = torch.abs(avg_similarity - best_id_avg_similarity)
#                 if similarity_difference <= 0.1 * best_id_avg_similarity:  # Threshold for similarity difference
#                     # Include only if the current ID is in the neighborhood of the best ID
#                     id_mask_np = id_mask.cpu().numpy()
#                     if np.logical_and(neighborhood_mask, id_mask_np).any():
#                         mask[id_mask] = 255

#         # Ensure best ID is always included
#         # mask = torch.zeros_like(logits_tensor, dtype=torch.uint8)
#         # mask[best_id_mask] = 255
#         mask[best_id_mask] = 255
#         # print(f"Best ID: {best_id}, Best ID Mask Sum: {best_id_mask.sum().item()}")


#         # Convert to numpy for saving
#         mask_np = mask.cpu().numpy()

#         # Save mask as PNG
#         mask_filename = os.path.join(save_path, f"{input_text}.png")
#         mask_image = Image.fromarray(mask_np.astype(np.uint8))
#         mask_image.save(mask_filename)
#         print(f"Saved mask for '{input_text}' in {mask_filename}")
#############################################################
# LeRF with Time Measurement
#############################################################
# import time
# import torch
# import numpy as np
# import os
# from PIL import Image
# import open_clip as clip

# # Set device to GPU or CPU
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load CLIP model
# model_load_start = time.time()
# model, _, _ = clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k", precision="fp32")
# model.eval()
# model = model.to(device)
# print(f"Model loading time: {time.time() - model_load_start:.2f}s")

# text_queries = [
#     #  "baseball", "dinosaur", "rabbit", "shrilling_chicken", "weaving basket", "wood wall"
#     # "banana","black leather shoe", "camera", "hand","red bag", "white sheet"
#     # "dressing_doll","green_grape","mini_offroad_car","orange_cat","pebbled_concrete_wall","Portuguese_egg_target","wood"
#     # "black_headphone","green_lawn","hand_soap","New_York_Yankees_cap","red_apple","stapler"
#     # "a red Nintendo Switch joy-con controller","a stack of UNO cards", " grey sofa", "Gundam","Pikachu","Xbox wireless controller"
#     # "chopsticks", "egg", "glass of water", "pork belly", "wavy noodles in bowl", "yellow bowl"
#     # "green apple", "green toy chair", "old camera", "porcelain hand", "red apple", "red toy chair", "rubber duck with red hat"
#     "apple", "bag of cookies", "coffee","coffee mug", "cookies on the plate", "napkin", "plate", "sheep","spoon handle", "stuffed bear","tea in a glass"
    
# ]

# # Data paths
# decoded_path = "/mnt/jsm/ILGS/output/lerf/teatime/test/ours_30000_text/feature_map_npy/decoded/"
# logits_path = "/mnt/jsm/ILGS/output/lerf/teatime/test/ours_30000_text/logits/"
# save_base_path = "/mnt/jsm/ILGS/output/lerf/teatime/test/ours_30000_text/test_mask/"

# # Neighborhood configuration
# d = 2
# neighbor_offsets = [(dy, dx) for dy in range(-d, d+1) for dx in range(-d, d+1) if not (dy==0 and dx==0)]

# def timeit(func):
#     """Decorator for timing functions"""
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         result = func(*args, **kwargs)
#         print(f"{func.__name__} executed in {time.time()-start:.4f}s")
#         return result
#     return wrapper

# @timeit
# def process_frame(idx):
#     """Process single frame with timing"""
#     # Load data
#     load_start = time.time()
#     decoded_feature_map = np.load(f"{decoded_path}decoded_{idx:05d}.npy")
#     logits = np.load(f"{logits_path}{idx:05d}_logits.npy")
#     H, W, C = decoded_feature_map.shape
#     print(f"\nFrame {idx} loaded in {time.time()-load_start:.4f}s")

#     # Convert to tensor
#     tensor_start = time.time()
#     decoded_feature_tensor = torch.from_numpy(decoded_feature_map).float().to(device)
#     logits_tensor = torch.from_numpy(logits).long().to(device)
#     decoded_feature_flattened = decoded_feature_tensor.view(-1, C)
#     decoded_feature_flattened = decoded_feature_flattened / decoded_feature_flattened.norm(dim=-1, keepdim=True)
#     print(f"Tensor conversion: {time.time()-tensor_start:.4f}s")

#     # Process queries
#     save_path = os.path.join(save_base_path, str(idx))
#     os.makedirs(save_path, exist_ok=True)

#     for input_text in text_queries:
#         query_start = time.time()
#         print(f"\nProcessing '{input_text}'...")

#         # Text encoding
#         text_encode_start = time.time()
#         text_tokens = clip.tokenize([input_text]).to(device)
#         with torch.no_grad():
#             text_feature = model.encode_text(text_tokens).float()
#         text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
#         text_encode_time = time.time() - text_encode_start

#         # Similarity calculation
#         sim_start = time.time()
#         cosine_similarity = torch.matmul(decoded_feature_flattened, text_feature.T).squeeze().view(H, W)
#         sim_time = time.time() - sim_start

#         # Mask generation
#         mask_gen_start = time.time()
#         id_values = torch.unique(logits_tensor)
#         id_avg_similarity = {}

#         # Calculate ID similarities
#         for id_value in id_values:
#             id_mask = (logits_tensor == id_value)
#             if id_mask.sum() == 0: continue
#             similarities = cosine_similarity.masked_select(id_mask)
#             if similarities.numel() > 0:
#                 top_80 = torch.topk(similarities, int(0.8*similarities.numel())).values
#                 id_avg_similarity[id_value.item()] = top_80.mean() if top_80.numel() > 0 else 0

#         # Find best ID
#         sorted_ids = sorted(id_avg_similarity.items(), key=lambda x: x[1], reverse=True)
#         best_id, best_avg = sorted_ids[0] if sorted_ids else (None, 0)

#         # Generate neighborhood mask
#         best_id_mask = (logits_tensor == best_id)
#         neighborhood_mask = np.zeros_like(best_id_mask.cpu().numpy(), dtype=bool)
#         for dy, dx in neighbor_offsets:
#             shifted = np.roll(best_id_mask.cpu().numpy(), (dy, dx), (0,1))
#             neighborhood_mask |= shifted

#         # Create final mask
#         mask = torch.zeros_like(logits_tensor, dtype=torch.uint8)
#         for id_value in id_values:
#             if id_value == best_id: continue
#             id_mask = (logits_tensor == id_value)
#             similarities = cosine_similarity.masked_select(id_mask)
            
#             if similarities.numel() > 0:
#                 top_100 = torch.topk(similarities, int(1*similarities.numel())).values
#                 avg = top_100.mean() if top_100.numel() > 0 else 0
#                 diff = torch.abs(avg - best_avg)
                
#                 if diff <= 0.1 * best_avg:
#                     id_mask_np = id_mask.cpu().numpy()
#                     if np.logical_and(neighborhood_mask, id_mask_np).any():
#                         mask[id_mask] = 255

#         mask[best_id_mask] = 255
#         mask_gen_time = time.time() - mask_gen_start

#         # Save mask
#         save_start = time.time()
#         mask_filename = os.path.join(save_path, f"{input_text}.png")
#         Image.fromarray(mask.cpu().numpy().astype(np.uint8)).save(mask_filename)
#         save_time = time.time() - save_start

#         # Print timings
#         total_time = time.time() - query_start
#         print(f"""Timing breakdown:
#         - Text encode: {text_encode_time:.4f}s ({1/text_encode_time:.1f} FPS)
#         - Similarity: {sim_time:.4f}s ({1/sim_time:.1f} FPS)
#         - Mask gen: {mask_gen_time:.4f}s ({1/mask_gen_time:.1f} FPS)
#         - Save: {save_time:.4f}s ({1/save_time:.1f} FPS)
#         Total: {total_time:.4f}s ({1/total_time:.1f} FPS)""")

# # Process all frames
# for idx in range(3):
#     process_frame(idx)

# print("\nAll processing completed!")

#########################################################################################################
#3D-OVS 이건 보류 
####################################################################################################

# import torch
# import numpy as np
# import os
# from PIL import Image
# import open_clip as clip

# # Set device to GPU or CPU
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load CLIP model
# model, _, _ = clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k", precision="fp32")
# model.eval()
# model = model.to(device)

# # Text query list
# text_queries = [
#    "baseball", "dinosaur", "rabbit", "shrilling_chicken", "weaving basket", "wood wall"
# ]

# # Data paths
# decoded_path = "/mnt/jsm/gaussian-grouping/output/3D-OVS_my/sofa/test/ours_30000_text/feature_map_npy/decoded/"
# logits_path = "/mnt/jsm/gaussian-grouping/output/3D-OVS_my/sofa/test/ours_30000_text/logits/"
# save_base_path = "/mnt/jsm/gaussian-grouping/output/3D-OVS_my/sofa/test/ours_30000_text/test_mask/"

# # Process each file (여기서는 6개의 파일)
# for idx in range(6):
#     # Load feature map
#     decoded_feature_map = np.load(f"{decoded_path}decoded_{idx:05d}.npy")  # Shape: (H, W, C)
#     logits = np.load(f"{logits_path}{idx:05d}_logits.npy")  # Shape: (H, W)
#     H, W, C = decoded_feature_map.shape

#     # Convert feature map and logits to tensors
#     decoded_feature_tensor = torch.from_numpy(decoded_feature_map).float().to(device)
#     logits_tensor = torch.from_numpy(logits).long().to(device)

#     # Normalize feature tensor
#     decoded_feature_flattened = decoded_feature_tensor.view(-1, C)
#     decoded_feature_flattened = decoded_feature_flattened / decoded_feature_flattened.norm(dim=-1, keepdim=True)

#     # Process each text query
#     for input_text in text_queries:
#         # Encode text query
#         text_tokens = clip.tokenize([input_text]).to(device)
#         with torch.no_grad():
#             text_feature = model.encode_text(text_tokens).float()
#         text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

#         # Cosine similarity calculation
#         cosine_similarity = torch.matmul(decoded_feature_flattened, text_feature.T).squeeze()
#         cosine_similarity = cosine_similarity.view(H, W)

#         # Calculate average similarity for each ID using top 80% similarities
#         id_values = torch.unique(logits_tensor)
#         id_avg_similarity = {}
#         for id_value in id_values:
#             id_mask = (logits_tensor == id_value)
#             if id_mask.sum() == 0:
#                 continue
#             similarities = cosine_similarity.masked_select(id_mask)
#             if similarities.numel() > 0:
#                 top_80_similarities = torch.topk(similarities, int(0.8 * similarities.numel())).values
#                 avg_similarity = top_80_similarities.mean() if top_80_similarities.numel() > 0 else 0
#                 id_avg_similarity[id_value.item()] = avg_similarity

#         # Get the ID with the highest average similarity (best ID)
#         sorted_ids = sorted(id_avg_similarity.items(), key=lambda x: x[1], reverse=True)
#         best_id = sorted_ids[0][0]
#         best_id_avg_similarity = id_avg_similarity[best_id]

#         # Generate binary mask based on similarity comparison with best ID
#         mask = torch.zeros_like(logits_tensor, dtype=torch.uint8)
#         for id_value in id_values:
#             id_mask = (logits_tensor == id_value)
#             if id_mask.sum() == 0:
#                 continue
#             # Best ID 바로 적용
#             if id_value == best_id:
#                 mask[id_mask] = 255
#                 continue

#             # For other IDs: 전체 유사도를 사용하여 평균 계산 후, best ID와 10% 이하 차이면 mask에 포함
#             similarities = cosine_similarity.masked_select(id_mask)
#             if similarities.numel() > 0:
#                 top_similarities = torch.topk(similarities, int(0.7 * similarities.numel())).values
#                 avg_similarity = top_similarities.mean() if top_similarities.numel() > 0 else 0
#                 similarity_difference = torch.abs(avg_similarity - best_id_avg_similarity)
#                 if similarity_difference <= 0.1 * best_id_avg_similarity:
#                     mask[id_mask] = 255

#         # Save mask as PNG (불러오고 저장하는 부분 그대로)
#         mask_np = mask.cpu().numpy()
#         save_path = os.path.join(save_base_path, str(idx))
#         os.makedirs(save_path, exist_ok=True)
#         mask_filename = os.path.join(save_path, f"{input_text}.png")
#         mask_image = Image.fromarray(mask_np.astype(np.uint8))
#         mask_image.save(mask_filename)
#         print(f"Saved mask for '{input_text}' in {mask_filename}")

#########################################################################################################
#iteration.
#########################################################################################################
import torch
import numpy as np
import os
from PIL import Image
import open_clip as clip

# Set device to GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model, _, _ = clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k", precision="fp32")
model.eval()
model = model.to(device)

# 예시 text query
# Text query list
text_queries = [
    #  "baseball", "dinosaur", "rabbit", "shrilling_chicken", "weaving basket", "wood wall"
    # "banana","black leather shoe", "camera", "hand","red bag", "white sheet"
    # "dressing_doll","green_grape","mini_offroad_car","orange_cat","pebbled_concrete_wall","Portuguese_egg_target","wood"
    # "black_headphone","green_lawn","hand_soap","New_York_Yankees_cap","red_apple","stapler"
    # "a red Nintendo Switch joy-con controller","a stack of UNO cards", " grey sofa", "Gundam","Pikachu","Xbox wireless controller"
    # "chopsticks", "egg", "glass of water", "pork belly", "wavy noodles in bowl", "yellow bowl"
    "green apple", "green toy chair", "old camera", "porcelain hand", "red apple", "red toy chair", "rubber duck with red hat"
    # "apple", "bag of cookies", "coffee","coffee mug", "cookies on the plate", "napkin", "plate", "sheep","spoon handle", "stuffed bear","tea in a glass"
    
]

# Data paths
decoded_path = "/mnt/jsm/ILGS/output/lerf/teatime/test/ours_30000_text/feature_map_npy/decoded/"
logits_path = "/mnt/jsm/ILGS/output/lerf/teatime/test/ours_30000_text/logits/"
save_base_path = "/mnt/jsm/ILGS/output/lerf/teatime/test/ours_30000_text/test_mask/"

# 거리 d 이내의 모든 픽셀 이웃
d = 1
neighbor_offsets = [(dy, dx) 
                    for dy in range(-d, d + 1) 
                    for dx in range(-d, d + 1) 
                    if not (dy == 0 and dx == 0)]

for idx in range(6):
    # Load feature map & logits
    decoded_feature_map = np.load(f"{decoded_path}decoded_{idx:05d}.npy")  # (H, W, C)
    logits = np.load(f"{logits_path}{idx:05d}_logits.npy")                # (H, W)
    H, W, C = decoded_feature_map.shape

    decoded_feature_tensor = torch.from_numpy(decoded_feature_map).float().to(device)
    logits_tensor = torch.from_numpy(logits).long().to(device)

    # Normalize feature tensor (pixel-wise)
    decoded_feature_flattened = decoded_feature_tensor.view(-1, C)
    decoded_feature_flattened = decoded_feature_flattened / decoded_feature_flattened.norm(dim=-1, keepdim=True)
    
    # 유니크한 ID 목록
    id_values = torch.unique(logits_tensor)
    
    # ID별 평균 유사도를 구하기 위해, 우선 text query 마다 진행
    save_path = os.path.join(save_base_path, str(idx))
    os.makedirs(save_path, exist_ok=True)

    for input_text in text_queries:
        # Text encode
        text_tokens = clip.tokenize([input_text]).to(device)
        with torch.no_grad():
            text_feature = model.encode_text(text_tokens).float()
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)  # (1, C)

        # Cosine similarity map
        cosine_similarity = torch.matmul(decoded_feature_flattened, text_feature.T).squeeze()  # (H*W,)
        cosine_similarity = cosine_similarity.view(H, W)  # (H, W)

        # ------------------------------
        # 1) 모든 ID에 대해 평균 유사도 계산 (top-80%로 제한 가능)
        # ------------------------------
        id_avg_similarity = {}
        for id_value in id_values:
            mask_id = (logits_tensor == id_value)
            if mask_id.sum() == 0:
                continue
            sim_vals = cosine_similarity[mask_id]
            if sim_vals.numel() > 0:
                # 상위 80% 유사도만 추려 평균
                top_k = int(0.8 * sim_vals.numel())
                top_80 = torch.topk(sim_vals, top_k).values if top_k > 0 else sim_vals
                avg_sim = top_80.mean() if top_80.numel() > 0 else 0
                id_avg_similarity[id_value.item()] = avg_sim
            else:
                id_avg_similarity[id_value.item()] = 0

        if len(id_avg_similarity) == 0:
            # 유효한 ID가 없는 경우 처리
            continue

        # ------------------------------
        # 2) ID 간 인접관계(그래프) 만들기
        #    - 두 ID가 인접하려면 픽셀 레벨에서 서로 거리가 d 이내인 픽셀이 하나라도 있어야 함
        # ------------------------------
        # 먼저 ID마다 mask를 numpy로 빼놓는다
        id_to_mask = {}
        for id_value in id_values:
            id_to_mask[id_value.item()] = (logits_tensor == id_value).cpu().numpy()

        adjacency = {id_val.item(): set() for id_val in id_values}

        # 모든 ID 쌍에 대해 인접성 확인(단순 O(N^2) 방식)
        unique_ids = list(id_values.cpu().numpy())
        for i in range(len(unique_ids)):
            id_a = unique_ids[i]
            mask_a = id_to_mask[id_a]
            for j in range(i + 1, len(unique_ids)):
                id_b = unique_ids[j]
                mask_b = id_to_mask[id_b]
                # 인접 판별 (mask_a와 mask_b가 d=2 이내로 붙어있는 픽셀이 하나라도 있는지)
                # 실제로는 더 효율적인 방법들이 있지만, 여기서는 직관적으로 구현
                found_neighbor = False
                for dy, dx in neighbor_offsets:
                    # mask_a를 (dy, dx)만큼 시프트시킨 결과와 mask_b가 겹치는 곳이 있는지
                    shifted_a = np.roll(mask_a, shift=(dy, dx), axis=(0, 1))
                    if np.logical_and(shifted_a, mask_b).any():
                        found_neighbor = True
                        break
                if found_neighbor:
                    adjacency[id_a].add(id_b)
                    adjacency[id_b].add(id_a)

        # ------------------------------
        # 3) Region Growing (BFS)
        #    - 초기 seed: 평균 유사도가 가장 높은 ID
        #    - 큐에서 하나씩 꺼내, 인접 ID와의 유사도 비교 후 조건 만족 시 확산
        # ------------------------------
        # 가장 높은 평균 유사도를 가진 ID 찾기
        sorted_ids = sorted(id_avg_similarity.items(), key=lambda x: x[1], reverse=True)
        best_id = sorted_ids[0][0]  # seed

        visited = set()
        queue = [best_id]

        # 예: threshold = 0.1 (새 seed의 avg_sim을 기준으로 인접 ID의 avg_sim 비교)
        THRESHOLD_RATIO = 0.1

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            current_sim = id_avg_similarity[current_id]

            # current_id의 이웃을 순회하며 조건 만족 시 확산
            for neighbor_id in adjacency[current_id]:
                if neighbor_id in visited:
                    continue
                neighbor_sim = id_avg_similarity[neighbor_id]
                # 유사도 차이가 current_sim 기준으로 10% 이내인지 확인
                if current_sim > 0:
                    diff = abs(neighbor_sim - current_sim)
                    if diff <= THRESHOLD_RATIO * current_sim:
                        # neighbor_id도 확산 영역에 포함시키고 큐에 추가
                        queue.append(neighbor_id)
                else:
                    # current_sim == 0 인 경우는 특별 케이스
                    # neighbor_sim이 0이거나 아주 작은 경우 포함시킬지 등 정책 결정
                    if neighbor_sim == 0:
                        queue.append(neighbor_id)

        # ------------------------------
        # 4) visited(확산된 ID)만 최종 mask에 포함
        # ------------------------------
        mask = torch.zeros_like(logits_tensor, dtype=torch.uint8)
        for id_val in visited:
            mask[logits_tensor == id_val] = 255

        # Save mask as PNG
        mask_np = mask.cpu().numpy()
        mask_filename = os.path.join(save_path, f"{input_text}.png")
        mask_image = Image.fromarray(mask_np.astype(np.uint8))
        mask_image.save(mask_filename)
        print(f"Saved region-growing mask for '{input_text}' in {mask_filename}")

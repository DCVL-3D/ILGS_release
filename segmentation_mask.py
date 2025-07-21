import torch
import numpy as np
import os
from PIL import Image
import open_clip as clip
from argparse import ArgumentParser

# --- [수정된 부분 1] ---
# 데이터셋별 텍스트 쿼리를 딕셔너리로 관리
DATASET_TEXT_QUERIES = {
    "ramen": [
        "chopsticks", "egg", "glass of water", "pork belly", "wavy noodles in bowl", "yellow bowl"
    ],
    "figurines": [
        "green apple", "green toy chair", "old camera", "porcelain hand", "red apple", "red toy chair", "rubber duck with red hat"
    ],
    "teatime": [
        "apple", "bag of cookies", "coffee", "coffee mug", "cookies on the plate", "napkin", "plate", "sheep", "spoon handle", "stuffed bear", "tea in a glass"
    ],
    # 다른 데이터셋이 있다면 여기에 추가
}

def get_queries_for_dataset(model_path):
    """
    모델 경로에 포함된 데이터셋 이름을 기반으로 적절한 텍스트 쿼리 목록을 반환합니다.
    """
    for dataset_name, queries in DATASET_TEXT_QUERIES.items():
        if dataset_name in model_path:
            print(f"✅ Dataset '{dataset_name}' detected. Using the corresponding text queries.")
            return queries
    print("⚠️ Warning: Could not detect a known dataset name in the model path. No text queries selected.")
    return None
# --- [수정 끝] ---


def generate_masks(model_path, text_queries):
    """
    주어진 경로와 텍스트 쿼리를 사용하여 테스트 마스크를 생성하고 저장합니다.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- [수정된 부분 2] ---
    # 경로를 model_path 기반으로 동적 생성
    base_path = os.path.join(model_path, "test/ours_30000_text/")
    decoded_path = os.path.join(base_path, "feature_map_npy/decoded/")
    logits_path = os.path.join(base_path, "logits/")
    save_base_path = os.path.join(base_path, "test_mask/")
    # --- [수정 끝] ---

    # CLIP 모델 로드
    model, _, _ = clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k", precision="fp32")
    model.eval()
    model = model.to(device)

    # 이웃 픽셀 정의 (거리 d=1)
    d = 1
    neighbor_offsets = [(dy, dx) for dy in range(-d, d + 1) for dx in range(-d, d + 1) if not (dy == 0 and dx == 0)]

    for idx in range(6): # 이미지 인덱스 루프
        print(f"\n--- Processing image index: {idx} ---")
        try:
            decoded_feature_map = np.load(f"{decoded_path}decoded_{idx:05d}.npy")
            logits = np.load(f"{logits_path}{idx:05d}_logits.npy")
        except FileNotFoundError:
            print(f"Warning: Files for index {idx} not found. Skipping.")
            continue

        H, W, C = decoded_feature_map.shape
        decoded_feature_tensor = torch.from_numpy(decoded_feature_map).float().to(device)
        logits_tensor = torch.from_numpy(logits).long().to(device)

        decoded_feature_flattened = decoded_feature_tensor.view(-1, C)
        decoded_feature_flattened = decoded_feature_flattened / decoded_feature_flattened.norm(dim=-1, keepdim=True)
        
        id_values = torch.unique(logits_tensor)
        
        save_path = os.path.join(save_base_path, str(idx))
        os.makedirs(save_path, exist_ok=True)

        for input_text in text_queries:
            # 텍스트 인코딩
            text_tokens = clip.tokenize([input_text]).to(device)
            with torch.no_grad():
                text_feature = model.encode_text(text_tokens).float()
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

            # 코사인 유사도 맵 계산
            cosine_similarity = torch.matmul(decoded_feature_flattened, text_feature.T).squeeze().view(H, W)

            # 1) ID별 평균 유사도 계산
            id_avg_similarity = {}
            for id_value in id_values:
                mask_id = (logits_tensor == id_value)
                if mask_id.sum() == 0: continue
                sim_vals = cosine_similarity[mask_id]
                if sim_vals.numel() > 0:
                    top_k = int(0.7 * sim_vals.numel())
                    top_80 = torch.topk(sim_vals, top_k).values if top_k > 0 else sim_vals
                    id_avg_similarity[id_value.item()] = top_80.mean().item() if top_80.numel() > 0 else 0
                else:
                    id_avg_similarity[id_value.item()] = 0
            
            if not id_avg_similarity: continue

            # 2) ID 간 인접 그래프 생성
            id_to_mask = {val.item(): (logits_tensor == val).cpu().numpy() for val in id_values}
            adjacency = {val.item(): set() for val in id_values}
            unique_ids = [val.item() for val in id_values]

            for i in range(len(unique_ids)):
                id_a = unique_ids[i]
                mask_a = id_to_mask[id_a]
                for j in range(i + 1, len(unique_ids)):
                    id_b = unique_ids[j]
                    mask_b = id_to_mask[id_b]
                    found_neighbor = False
                    for dy, dx in neighbor_offsets:
                        shifted_a = np.roll(mask_a, shift=(dy, dx), axis=(0, 1))
                        if np.logical_and(shifted_a, mask_b).any():
                            found_neighbor = True
                            break
                    if found_neighbor:
                        adjacency[id_a].add(id_b)
                        adjacency[id_b].add(id_a)
            
            # 3) Region Growing (BFS)
            sorted_ids = sorted(id_avg_similarity.items(), key=lambda x: x[1], reverse=True)
            best_id = sorted_ids[0][0]

            visited = set()
            queue = [best_id]
            THRESHOLD_RATIO = 0.1

            while queue:
                current_id = queue.pop(0)
                if current_id in visited: continue
                visited.add(current_id)
                current_sim = id_avg_similarity[current_id]

                for neighbor_id in adjacency[current_id]:
                    if neighbor_id not in visited:
                        neighbor_sim = id_avg_similarity[neighbor_id]
                        if current_sim > 0 and abs(neighbor_sim - current_sim) <= THRESHOLD_RATIO * current_sim:
                            queue.append(neighbor_id)
                        elif current_sim == 0 and neighbor_sim == 0:
                            queue.append(neighbor_id)
            
            # 4) 최종 마스크 생성 및 저장
            mask = torch.zeros_like(logits_tensor, dtype=torch.uint8)
            for id_val in visited:
                mask[logits_tensor == id_val] = 255

            mask_np = mask.cpu().numpy()
            mask_filename = os.path.join(save_path, f"{input_text.replace(' ', '_')}.png")
            Image.fromarray(mask_np.astype(np.uint8)).save(mask_filename)
            print(f"  Saved mask for '{input_text}'")

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate test masks using region growing based on CLIP similarity.")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the model output directory (e.g., output/lerf/teatime)")
    args = parser.parse_args()

    # 모델 경로에서 데이터셋에 맞는 텍스트 쿼리 목록 가져오기
    queries = get_queries_for_dataset(args.model_path)
    
    if queries:
        # 마스크 생성 함수 호출
        generate_masks(args.model_path, queries)
        print("\nMask generation complete.")
    else:
        print("Error: Could not determine which text queries to use. Please check the model path.")

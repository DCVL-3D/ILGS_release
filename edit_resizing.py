import torch
import numpy as np
import open_clip as clip
from scene import Scene
from gaussian_renderer import GaussianModel
from tqdm import tqdm
import os
from os import makedirs
import torchvision
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from argparse import ArgumentParser
from PIL import Image
from scipy.spatial import Delaunay
from gaussian_renderer import render

def find_best_id(model_path, text_query, device):

    print("Finding best matching ID for the object...")
    
    decoded_path = os.path.join(model_path, "test/ours_30000_text/feature_map_npy/decoded/")
    logits_path = os.path.join(model_path, "test/ours_30000_text/logits/")

    model, _, _ = clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k", precision="fp32")
    model.eval()
    model = model.to(device)

    best_id = None
    for idx in range(1):
        decoded_feature_map_path = os.path.join(decoded_path, f"decoded_{idx:05d}.npy")
        logits_map_path = os.path.join(logits_path, f"{idx:05d}_logits.npy")

        if not os.path.exists(decoded_feature_map_path) or not os.path.exists(logits_map_path):
            print(f"Error: Feature or logits file not found for index {idx}.")
            return None

        decoded_feature_map = np.load(decoded_feature_map_path)
        logits = np.load(logits_map_path)
        H, W, C = decoded_feature_map.shape

        decoded_feature_tensor = torch.from_numpy(decoded_feature_map).float().to(device)
        logits_tensor = torch.from_numpy(logits).long().to(device)

        decoded_feature_flattened = decoded_feature_tensor.view(-1, C)
        decoded_feature_flattened = decoded_feature_flattened / decoded_feature_flattened.norm(dim=-1, keepdim=True)

        text_tokens = clip.tokenize([text_query]).to(device)
        with torch.no_grad():
            text_feature = model.encode_text(text_tokens).float()
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

        cosine_similarity = torch.matmul(decoded_feature_flattened, text_feature.T).squeeze().view(H, W)

        id_values = torch.unique(logits_tensor)
        id_avg_similarity = {}
        for id_value in id_values:
            id_mask = (logits_tensor == id_value)
            if id_mask.sum() == 0: continue
            similarities = cosine_similarity.masked_select(id_mask)
            if similarities.numel() > 0:
                top_k_count = max(1, int(0.8 * similarities.numel()))
                top_80_similarities = torch.topk(similarities, top_k_count).values
                id_avg_similarity[id_value.item()] = top_80_similarities.mean()

        if not id_avg_similarity:
            print("Could not calculate similarities for any ID.")
            continue
            
        sorted_ids = sorted(id_avg_similarity.items(), key=lambda x: x[1], reverse=True)
        best_id = sorted_ids[0][0]

    return best_id

def get_scale_factor_from_user():

    while True:
        try:
            scale_str = input("Enter the scale factor (e.g., 0.5 to shrink, 1.5 to enlarge): ")
            scale_factor = float(scale_str)
            if scale_factor > 0:
                return scale_factor
            else:
                print("Scale factor must be a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number (e.g., 0.5).")

def modify_gaussian_size(gaussians, classifier, selected_obj_id, scale_factor):

    with torch.no_grad():
        logits3d = classifier(gaussians._objects_dc.permute(2, 0, 1))
        prob_obj3d = torch.softmax(logits3d, dim=0)
        mask3d = prob_obj3d[selected_obj_id, :, :].squeeze() > 0.3

        if mask3d.sum() == 0:
            print("Warning: No Gaussians found for the selected object ID above the threshold.")
            return gaussians

        object_xyz = gaussians.get_xyz[mask3d]
        object_center = object_xyz.mean(dim=0, keepdim=True)

        gaussians._xyz[mask3d] = object_center + (gaussians._xyz[mask3d] - object_center) * scale_factor

        scaling = gaussians.get_scaling
        gaussians._scaling[mask3d] = gaussians.scaling_inverse_activation(
            scaling[mask3d] * scale_factor
        )
    return gaussians

def render_set(model_path, name, iteration_str, views, gaussians, pipeline, background):

    render_path = os.path.join(model_path, name, f"ours{iteration_str}", "renders_resized")
    makedirs(render_path, exist_ok=True)
    
    print(f"📏 Rendering resized results and saving to: {render_path}")
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        with torch.no_grad():
            results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, f'{idx:05d}.png'))
    print("Rendering complete.")

def modify(dataset, iteration, pipeline, skip_train, skip_test):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    text_query = input("Enter the text query of the object to resize (e.g., 'a tea cup'): ")
    best_id = find_best_id(dataset.model_path, text_query, device)

    if best_id is None:
        print("Could not find a matching object. Exiting.")
        return
    print(f"✅ Best matching ID for '{text_query}' is {best_id}.")
    
    scale_factor = get_scale_factor_from_user()

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    classifier_path = os.path.join(dataset.model_path, "point_cloud", f"iteration_{scene.loaded_iter}", "classifier.pth")
    if not os.path.exists(classifier_path):
        print(f"Error: Classifier file not found at {classifier_path}")
        return

    classifier = torch.nn.Conv2d(gaussians.num_objects, 256, kernel_size=1).to(device)
    classifier.load_state_dict(torch.load(classifier_path))
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    gaussians = modify_gaussian_size(gaussians, classifier, best_id, scale_factor)

    iteration_str = f"_object_resize/iteration_{scene.loaded_iter}"

    if not skip_train:
        render_set(dataset.model_path, "train", iteration_str, scene.getTrainCameras(), gaussians, pipeline, background)
    if not skip_test:
        render_set(dataset.model_path, "test", iteration_str, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    parser = ArgumentParser(description="Object Resizing Script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    safe_state(args.quiet)
    
    modify(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

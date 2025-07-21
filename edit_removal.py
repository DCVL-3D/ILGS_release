import torch
import numpy as np
import open_clip as clip
import json
from scene import Scene
from gaussian_renderer import GaussianModel
from tqdm import tqdm
import os
from os import makedirs
import torchvision
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from argparse import ArgumentParser
from render import feature_to_rgb, visualize_obj
from PIL import Image
import cv2
from scipy.spatial import Delaunay
from gaussian_renderer import render
import colorsys
from sklearn.decomposition import PCA


def find_best_id(model_path, text_query, device):
 
    print("Finding best matching ID...")
    
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
            print(f"Looked for: {decoded_feature_map_path}")
            print(f"And: {logits_map_path}")
            return None

        decoded_feature_map = np.load(decoded_feature_map_path)  # Shape: (H, W, C)
        logits = np.load(logits_map_path)  # Shape: (H, W)
        H, W, C = decoded_feature_map.shape

        decoded_feature_tensor = torch.from_numpy(decoded_feature_map).float().to(device)
        logits_tensor = torch.from_numpy(logits).long().to(device)

        decoded_feature_flattened = decoded_feature_tensor.view(-1, C)
        decoded_feature_flattened = decoded_feature_flattened / decoded_feature_flattened.norm(dim=-1, keepdim=True)

        text_tokens = clip.tokenize([text_query]).to(device)
        with torch.no_grad():
            text_feature = model.encode_text(text_tokens).float()
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

        cosine_similarity = torch.matmul(decoded_feature_flattened, text_feature.T).squeeze()
        cosine_similarity = cosine_similarity.view(H, W)

        id_values = torch.unique(logits_tensor)
        id_avg_similarity = {}
        for id_value in id_values:
            id_mask = (logits_tensor == id_value)
            if id_mask.sum() == 0:
                continue
            similarities = cosine_similarity.masked_select(id_mask)
            if similarities.numel() > 0:
                top_80_count = int(0.8 * similarities.numel())
                if top_80_count == 0 and similarities.numel() > 0:
                    top_80_count = 1 
                
                top_80_similarities = torch.topk(similarities, top_80_count).values
                avg_similarity = top_80_similarities.mean() if top_80_similarities.numel() > 0 else 0
                id_avg_similarity[id_value.item()] = avg_similarity

        if not id_avg_similarity:
            print("Could not calculate similarities for any ID.")
            continue
            
        sorted_ids = sorted(id_avg_similarity.items(), key=lambda x: x[1], reverse=True)
        best_id = sorted_ids[0][0]

    return best_id


def points_inside_convex_hull(point_cloud, mask, remove_outliers=True, outlier_factor=1.0):

    masked_points = point_cloud[mask].cpu().numpy()
    if masked_points.shape[0] < 4: 
        return torch.zeros(point_cloud.shape[0], dtype=torch.bool, device='cuda')

    if remove_outliers:
        Q1 = np.percentile(masked_points, 25, axis=0)
        Q3 = np.percentile(masked_points, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR))
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
        if filtered_masked_points.shape[0] < 4: 
             filtered_masked_points = masked_points
    else:
        filtered_masked_points = masked_points
    
    if filtered_masked_points.shape[0] < 4:
        return torch.zeros(point_cloud.shape[0], dtype=torch.bool, device='cuda')

    delaunay = Delaunay(filtered_masked_points)
    points_inside_hull_mask = delaunay.find_simplex(point_cloud.cpu().numpy()) >= 0
    return torch.tensor(points_inside_hull_mask, device='cuda')

def removal_setup(opt, model_path, iteration, views, gaussians, pipeline, background, classifier, selected_obj_ids, cameras_extent):
    removal_thresh = 0.3
    selected_obj_ids = torch.tensor(selected_obj_ids).cuda()
    with torch.no_grad():
        logits3d = classifier(gaussians._objects_dc.permute(2, 0, 1))
        prob_obj3d = torch.softmax(logits3d, dim=0)
        mask = prob_obj3d[selected_obj_ids, :, :] > removal_thresh
        mask3d = mask.any(dim=0).squeeze()

        mask3d_convex = points_inside_convex_hull(gaussians._xyz.detach(), mask3d, outlier_factor=1.0)
        mask3d = torch.logical_or(mask3d, mask3d_convex)

        mask3d = mask3d.float()[:, None, None]

    gaussians.removal_setup(opt, mask3d)

    point_cloud_path = os.path.join(model_path, "point_cloud_object_removal/iteration_{}".format(iteration))
    makedirs(point_cloud_path, exist_ok=True)
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    return gaussians


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):

    render_path = os.path.join(model_path, name, "ours{}".format(iteration), "renders_removed")
    makedirs(render_path, exist_ok=True)
    
    print(f"Rendering results and saving to: {render_path}")

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        with torch.no_grad():
            results = render(view, gaussians, pipeline, background)
            rendering = results["render"]
            
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

    print("Rendering complete.")



def removal(dataset, iteration, pipeline, skip_train, skip_test, opt):
    num_classes = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_query = input("Enter the text query to remove (e.g., 'the yellow dog'): ")
    
    best_id = find_best_id(dataset.model_path, text_query, device)

    if best_id is None:
        print("Could not find a matching object. Exiting.")
        return
        
    print(f"✅ Best matching ID for '{text_query}' is {best_id}. Proceeding with removal.")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    classifier_path = os.path.join(dataset.model_path, "point_cloud", f"iteration_{scene.loaded_iter}", "classifier.pth")
    if not os.path.exists(classifier_path):
        print(f"Error: Classifier file not found at {classifier_path}")
        return

    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    classifier.cuda()
    classifier.load_state_dict(torch.load(classifier_path))
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = removal_setup(opt, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, [best_id], scene.cameras_extent)

    scene = Scene(dataset, gaussians, load_iteration='_object_removal/iteration_' + str(scene.loaded_iter), shuffle=False)
    
    if not skip_train:
        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
    if not skip_test:
        render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int) 
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    safe_state(args.quiet)

    removal(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, opt.extract(args))
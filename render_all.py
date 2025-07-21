import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, GaussianModel
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import numpy as np
from PIL import Image
from render import feature_to_rgb, visualize_obj

def render_set_all(model_path, name, iteration, views, gaussians, pipeline, background, classifier):

    base_path = os.path.join(model_path, name, "ours_{}_text".format(iteration))
    renders_path = os.path.join(base_path, "renders")
    colormask_path = os.path.join(base_path, "objects_feature16")
    feature_npy_path = os.path.join(base_path, "feature_map_npy")
    logits_path = os.path.join(base_path, "logits")
    
    for path in [renders_path, colormask_path, feature_npy_path, logits_path]:
        makedirs(path, exist_ok=True)
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        rendering_clip = results["feature_map"]

        logits = classifier(rendering_obj)

        np.save(os.path.join(feature_npy_path, '{0:05d}.npy'.format(idx)), rendering_clip.cpu().numpy())

        torchvision.utils.save_image(rendering, os.path.join(renders_path, '{0:05d}.png'.format(idx)))

        rgb_mask = feature_to_rgb(rendering_obj)
        Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{0:05d}.png'.format(idx)))

        logits_np = logits.cpu().numpy()
        logits_single_channel = np.argmax(logits_np, axis=0)
        np.save(os.path.join(logits_path, '{0:05d}_logits.npy'.format(idx)), logits_single_channel)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        dataset.eval = True
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        num_classes = dataset.num_classes
        print("Num classes: ", num_classes)
        
        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier_path = os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(scene.loaded_iter), "classifier.pth")
        classifier.load_state_dict(torch.load(classifier_path))
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        for phase, skip in [("train", skip_train), ("test", skip_test)]:
            if not skip:
                if phase == "train":
                    cams = scene.getTrainCameras()
                else:
                    cams = scene.getTestCameras()
                render_set_all(dataset.model_path, phase, scene.loaded_iter, cams, gaussians, pipeline, background, classifier)

if __name__ == "__main__":
    parser = ArgumentParser(description="Unified Rendering Script: Creates Rendering Images, Feature Maps, and ID Maps")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    
    print("Rendering " + args.model_path)
    
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

import cv2
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
import pytorch_lightning as pl
pl.seed_everything(42)

# TODO: figure out why this is necessary
import sys
src_demo_path = sys.path[0]
src_path = src_demo_path[:-5]
project_path = src_path[:-3]
sys.path.insert(0, src_path)
sys.path.insert(0, project_path)

import torch

from models.autoencoder import Autoencoder
from modules.planners import DifferentiableDiagAstar

def load_image_tensor(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    tensor = torch.tensor(image, dtype = torch.float32).unsqueeze(0).unsqueeze(0)
    tensor[tensor == 255] = 1
    return tensor

def transform_plan(image):
    result = torch.ones_like(image)
    result[image[:, :, 0] == 1] = torch.tensor([0.1, 0.1, 0.1])
    result[image[:, :, 1] == 1] = torch.tensor([1., 0., 0])
    result[image[:, :, 2] == 1] = torch.tensor([0., 1, 0.])
    return result

def infer_path(
    pathfinding_method = 'cf', 
    resolution = (64, 64),
    goal_path = 'goal_map.png', 
    map_path = 'rescaled_map.png', 
    start_path = 'start_map.png', 
    weights_path = 'cf.pth'
):
    goal_filepath = os.path.join(CURRENT_DIR, '..', '..', 'map_data', goal_path)
    map_filepath = os.path.join(CURRENT_DIR, '..', '..', 'map_data', map_path)
    start_filepath = os.path.join(CURRENT_DIR, '..', '..', 'map_data', start_path)
    weights_filepath = os.path.join(CURRENT_DIR, '..', 'weights', weights_path)

    goal = load_image_tensor(goal_filepath)
    map_design = load_image_tensor(map_filepath)
    start = load_image_tensor(start_filepath)
    weights = torch.load(weights_filepath)

    inputs_g = torch.cat([map_design, goal], dim=1)
    inputs_sg = torch.cat([map_design, start + goal], dim=1)

    planner = None
    model = None

    if pathfinding_method == 'f':
        planner = DifferentiableDiagAstar(mode =' f')
        model_focal = Autoencoder(mode='f', resolution = resolution)
        model_focal.load_state_dict(weights)
        model_focal.eval()
        model = model_focal
    elif pathfinding_method == 'fw100':
        planner = DifferentiableDiagAstar(mode = 'f', f_w = 100)
        model_focal = Autoencoder(mode='f', resolution = resolution)
        model_focal.load_state_dict(weights)
        model_focal.eval()
        model = model_focal
    elif pathfinding_method == 'cf':
        planner = DifferentiableDiagAstar(mode = 'k')
        model_cf = Autoencoder(mode = 'k', resolution = resolution)
        model_cf.load_state_dict(weights)
        model_cf.eval()
        model = model_cf
    elif pathfinding_method == 'w2':
        planner = DifferentiableDiagAstar(mode = 'default', h_w = 2)
    elif pathfinding_method == 'vanilla':
        planner = DifferentiableDiagAstar(mode = 'default', h_w = 1)
    else:
        raise ValueError("Invalid pathfinding_method value. Choose from 'f', 'fw100', 'cf', 'w2', 'vanilla'.")

    with torch.no_grad():
        if model:
            pred = (model(inputs_sg) + 1) / 2
        else:
            pred = (map_design == 0) * 1.

        if planner:
            outputs = planner(
                pred,
                start,
                goal,
                (map_design == 0) * 1.
            )
        else:
            outputs = None

    return {
        'map_design': map_design,
        'outputs': outputs,
        'prediction': pred
    }
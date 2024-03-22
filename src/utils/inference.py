
import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image
import pytorch_lightning as pl
import sys
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
pl.seed_everything(42)
src_dir = os.path.join(CURRENT_DIR, '..', '..', 'src')
map_dir = os.path.join(CURRENT_DIR, '..', '..', 'map_data')
sys.path.append(src_dir)
sys.path.append(map_dir)

from models.autoencoder import Autoencoder
from modules.planners import DifferentiableDiagAstar

def resize_image(image, resolution):
    img = Image.fromarray(image)
    original_width, original_height = img.size
    aspect_ratio = original_width / original_height

    if aspect_ratio > 1:
        new_width = resolution[0]
        new_height = round(new_width / aspect_ratio)
    else:
        new_height = resolution[1]
        new_width = round(new_height * aspect_ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    padded_img = Image.new("L", resolution, color="black")
    padded_img.paste(img, ((resolution[0] - new_width) // 2, (resolution[1] - new_height) // 2))
    padded_img = padded_img.point(lambda x: 1 if x > 0 else 0)
    return np.asarray(padded_img)

def load_image_tensor(file_path, resolution):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    image = resize_image(image, resolution)
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor

def transform_plan(image):
    result = torch.ones_like(image)
    result[image[:, :, 0] == 1] = torch.tensor([0.1, 0.1, 0.1])
    result[image[:, :, 1] == 1] = torch.tensor([1., 0., 0])
    result[image[:, :, 2] == 1] = torch.tensor([0., 1, 0.])
    return result

def infer_path(
    pathfinding_method = 'f',
    model_resolution = (64, 64),
    img_resolution = (512, 512),
    goal_path = 'example/mw/goal.png',
    map_path = 'example/mw/map.png',
    start_path = 'example/mw/start.png',
    weights_path = 'weights/focal.pth'
):
    goal = load_image_tensor(goal_path, resolution = img_resolution)
    map_design = load_image_tensor(map_path, resolution = img_resolution)
    start = load_image_tensor(start_path, resolution = img_resolution)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torch.load(weights_path, map_location = device)
    weights = weights['state_dict'] if Path(weights_path).suffix == '.ckpt' else weights

    model = None
    inputs = None

    if pathfinding_method in ['f', 'fw100']:
        inputs = torch.cat([map_design, start + goal], dim=1)
        model = Autoencoder(mode='f', resolution = model_resolution)
        model.load_state_dict(weights)

        _resolution = (img_resolution[0] // 2**3, img_resolution[1] // 2**3) # 3 is hardcoded downsample steps
        model.pos.change_resolution(_resolution, 1.)
        model.decoder_pos.change_resolution(_resolution, 1.)
        model.eval()

        if pathfinding_method == 'fw100':
            planner = DifferentiableDiagAstar(mode='f', f_w=100)
        else:
            planner = DifferentiableDiagAstar(mode=' f')

    elif pathfinding_method == 'cf':
        inputs = torch.cat([map_design, goal], dim=1)
        planner = DifferentiableDiagAstar(mode = 'k')
        model = Autoencoder(mode = 'k', resolution = model_resolution)
        model.load_state_dict(weights)
        _resolution = (img_resolution[0] // 2**3, img_resolution[1] // 2**3) # 3 is hardcoded downsample steps
        model.pos.change_resolution(_resolution, 1.)
        model.decoder_pos.change_resolution(_resolution, 1.)
        model.eval()

    elif pathfinding_method == 'w2':
        planner = DifferentiableDiagAstar(mode = 'default', h_w = 2)

    elif pathfinding_method == 'vanilla':
        planner = DifferentiableDiagAstar(mode = 'default', h_w = 1)

    else:
        raise ValueError("Invalid pathfinding_method value. Choose from 'f', 'fw100', 'cf', 'w2', 'vanilla'.")
    
    with torch.no_grad():
        if model:
            pred = (model(inputs) + 1) / 2
        else:
            pred = (map_design == 0) * 1.
        outputs = planner(
            pred,
            start,
            goal,
            (map_design == 0) * 1.
        )

    return {
        'map_design': map_design,
        'outputs': outputs,
        'prediction': pred
    }

if __name__ == "__main__":
    infer_path()
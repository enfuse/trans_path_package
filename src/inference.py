import os
import sys
from pathlib import Path

import cv2
import pytorch_lightning as pl
import torch
from models.autoencoder import Autoencoder
from modules.planners import DifferentiableDiagAstar
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
pl.seed_everything(42)


def resize_image(image, resolution):
    # Calculate aspect ratio of the original image
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height

    # Calculate new dimensions while maintaining aspect ratio
    target_width, target_height = resolution
    if target_width / aspect_ratio <= target_height:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    # Resize the image using the calculated dimensions
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # Calculate the coordinates to paste the resized image in the center of the canvas
    start_x = (target_width - new_width) // 2
    start_y = (target_height - new_height) // 2

    # Create a canvas with the target size and fill with zeros (black)
    canvas = np.zeros((target_height, target_width), dtype=np.uint8)

    # Paste the resized image onto the canvas
    canvas[start_y:start_y + new_height, start_x:start_x + new_width] = resized_image

    canvas[canvas > 0] = 1

    # Check if the white pixel is present in the resized image
    if np.max(canvas) == 0:
        # If the white pixel is lost, find the position of the white pixel in the original image
        white_pixel_position = np.argwhere(image == 255)[0]

        # Calculate the corresponding position in the resized image
        new_white_pixel_position = ((white_pixel_position *
                                    np.array([new_width / original_width, new_height / original_height]))
                                    .astype(int)
                                    )

        # Set the corresponding pixel in the resized canvas to white
        canvas[new_white_pixel_position[0], new_white_pixel_position[1]] = 255


    return canvas

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
        resolution = (256, 256),
        goal_path = 'example/goal.png',
        map_path = 'example/map.png',
        start_path = 'example/start.png',
        weights_path = 'weights/focal.pth'
):

    goal = load_image_tensor(goal_path, resolution=resolution)
    map_design = load_image_tensor(map_path, resolution=resolution)
    start = load_image_tensor(start_path, resolution=resolution)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torch.load(weights_path, map_location = device)

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

if __name__ == "__main__":
    infer_path()
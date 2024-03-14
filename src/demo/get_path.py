import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import sys
src_demo_path = sys.path[0]
src_path = src_demo_path[:-5]
project_path = src_path[:-3]
sys.path.insert(0, src_path)
sys.path.insert(0, project_path)
import torch
from utils import bw_map_data_generator as map_gen
from utils import inference as inf

def generate_map_with_path(results, file_path):
    data_cf =  torch.cat(
        [results['map_design'], results['outputs_cf'].paths, results['outputs_cf'].histories - results['outputs_cf'].paths], dim=1
    )
    np_data_cf = data_cf.numpy()
    image_data = np_data_cf.transpose(0, 2, 3, 1)
    scaled_image_data = (image_data * 255).astype(np.uint8)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    map_data_dir = os.path.join(script_dir, '..', '..', 'map_data')
    os.makedirs(map_data_dir, exist_ok=True)
    output_image_path = os.path.join(map_data_dir, 'output.png')
    cv2.imwrite(output_image_path, scaled_image_data[0])

# input params
def parse_args():
    #  Args to generate scaled map and start and end maps
    parser = argparse.ArgumentParser(description = "Path Planning Visualization Script")
    parser.add_argument("--image_path", type = str, help = "Path to unscaled map image")
    parser.add_argument("--target_size_x", type = int, default = 64, help = "Dictates size x(width) of image")
    parser.add_argument("--target_size_y", type = int, default = 64, help = "Dictates size y(height) of image")
    parser.add_argument("--start_point_x", type = int, help = "Starting x point for start image")
    parser.add_argument("--start_point_y", type = int, help = "Starting y point for start image")
    parser.add_argument("--goal_point_x", type = int, help = "Goal x point for goal image")
    parser.add_argument("--goal_point_y", type = int, help = "Goal y point for goal image")
    args = parser.parse_args()
    return args

def main(args):
    start_filename = 'start_map'
    goal_filename = 'goal_map'
    map_gen.resize_and_pad(
        image_path = args.image_path, 
        target_size_x = args.target_size_x, 
        target_size_y = args.target_size_y
    )
    map_gen.create_start_or_goal_image(
        x_point = args.start_point_x, 
        y_point = args.start_point_y, 
        filename = start_filename, 
        target_size_x = args.target_size_x, 
        target_size_y = args.target_size_y
    )
    map_gen.create_start_or_goal_image(
        x_point = args.goal_point_x, 
        y_point = args.goal_point_y, 
        filename = goal_filename, 
        target_size_x = args.target_size_x, 
        target_size_y = args.target_size_y
    )
    results = inf.infer_path()
    generate_map_with_path(results = results, file_path = 'wa_star_cf_image.png')
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
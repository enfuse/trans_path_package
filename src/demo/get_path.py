import argparse
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

    fig, ax = plt.subplots(1, 6, figsize=(15, 7))

    ax[0].imshow(
        inf.transform_plan(torch.moveaxis(torch.cat(
            [results['map_design'], results['outputs_vanilla'].paths, results['outputs_vanilla'].histories - results['outputs_vanilla'].paths], dim=1)[0], 0, 2)
                        ))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title(f'A* search')

    ax[1].imshow(
        inf.transform_plan(torch.moveaxis(torch.cat(
            [results['map_design'], results['outputs_w2'].paths, results['outputs_w2'].histories - results['outputs_w2'].paths], dim=1)[0], 0, 2)
                        ))
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title(f'WA* search')

    ax[2].imshow(
        inf.transform_plan(torch.moveaxis(torch.cat(
            [results['map_design'], results['outputs_fw100'].paths, results['outputs_fw100'].histories - results['outputs_fw100'].paths], dim=1)[0], 0, 2)
                        ))
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_title(f'Focal search + PPM')

    ax[3].imshow(
        inf.transform_plan(torch.moveaxis(torch.cat(
            [results['map_design'], results['outputs_cf'].paths, results['outputs_cf'].histories - results['outputs_cf'].paths], dim=1)[0], 0, 2)
                        ))
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].set_title(f'WA* search + CF')

    ax[4].imshow(results['pred_f'][0, 0], cmap='gray')
    ax[4].set_xticks([])
    ax[4].set_yticks([])
    ax[4].set_title(f'predicted path')

    ax[5].imshow(results['pred_cf'][0, 0])
    ax[5].set_xticks([])
    ax[5].set_yticks([])
    ax[5].set_title(f'predicted CF')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    main(args)
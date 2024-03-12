import argparse
import numpy as np
import os
from PIL import Image
import sys
sys.path.append('../..')
from src.utils import bw_map_data_generator as map_gen
from src.utils import inference as inf

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

    # Load map image
    map_image = Image.open("./../../map_data/rescaled_map.png")
    map_arr = np.array(map_image)

    # Overlay path onto the map_arr
    path = results['outputs_vanilla'].paths.squeeze().cpu().numpy()
    path = np.where(path > 0, 255, 0).astype(np.uint8)
    map_arr[:, :, 0] = np.where(path > 0, 255, map_arr[:, :, 0])  # Red channel
    map_arr[:, :, 1] = np.where(path > 0, 0, map_arr[:, :, 1])    # Green channel
    map_arr[:, :, 2] = np.where(path > 0, 0, map_arr[:, :, 2])    # Blue channel

    # Overlay start point
    start = np.zeros_like(path)
    start[args.start_point_y, args.start_point_x] = 255
    map_arr[:, :, 1] = np.where(start > 0, 255, map_arr[:, :, 1])  # Green channel

    # Overlay goal point
    goal = np.zeros_like(path)
    goal[args.goal_point_y, args.goal_point_x] = 255
    map_arr[:, :, 2] = np.where(goal > 0, 255, map_arr[:, :, 2])    # Blue channel

    # Save the result image
    result_image = Image.fromarray(map_arr)
    result_image.save("result_image.png")

if __name__ == "__main__":
    args = parse_args()
    main(args)
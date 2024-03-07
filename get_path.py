import argparse
import matplotlib.pyplot as plt
import inference
import bw_map_data_generator

# input params
def parse_args():
    parser = argparse.ArgumentParser(description="Path Planning Visualization Script")
    # Args to generate scaled map and start and end maps
    parser.add_argument("--image_path",
                        type=str,
                        help="Path to unscaled map image"
                        )
    parser.add_argument("--target_size_x",
                        type=int,
                        default=64,
                        help="Dictates size x(width) of image"
                        )
    parser.add_argument("--target_size_y",
                        type=int,
                        default=64,
                        help="Dictates size y(height) of image"
                        )
    parser.add_argument("--start_point_x",
                        type=int,
                        help="Starting x point for start image"
                        )
    parser.add_argument("--start_point_y",
                        type=int,
                        help="Starting y point for start image"
                        )
    parser.add_argument("--goal_point_x",
                        type=int,
                        help="Goal x point for goal image"
                        )
    parser.add_argument("--goal_point_y",
                        type=int,
                        help="Goal y point for goal image"
                        )

    args = parser.parse_args()

    return args

def main(args):
    start_filename = 'start_map'
    goal_filename = 'goal_map'
    # create resized map
    bw_map_data_generator.resize_and_pad(args.image_path, args.target_size_x, args.target_size_y)
    # create start and goal images
    bw_map_data_generator.create_start_or_goal_image(args.start_point_x, args.start_point_y, 
                                                     start_filename, args.target_size_x, args.target_size_y)
    bw_map_data_generator.create_start_or_goal_image(args.goal_point_x, args.goal_point_y, 
                                                     goal_filename, args.target_size_x, args.target_size_y)
    # infer path 
    inference.infer_path()

if __name__ == "__main__":
    args = parse_args()

    main(args)
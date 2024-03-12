import argparse
import cv2
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import os

from models.autoencoder import Autoencoder
from modules.planners import DifferentiableDiagAstar

# Set environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Get the directory path of the current script
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

pl.seed_everything(42)

'''
# input params
def parse_args():
    parser.add_argument("--device",
                        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the script on (default: 'cuda' if available, else 'cpu')"
                        )

    args = parser.parse_args()

    return args
'''

# convert image into binary value tensor?
def load_image_tensor(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tensor[tensor == 255] = 1

    return tensor


def transform_plan(image):
    result = torch.ones_like(image)
    result[image[:, :, 0] == 1] = torch.tensor([0.1, 0.1, 0.1])
    result[image[:, :, 1] == 1] = torch.tensor([1., 0., 0])
    result[image[:, :, 2] == 1] = torch.tensor([0., 1, 0.])

    return result


def infer_path():
    # Define paths
    cf_weight_path = os.path.join(CURRENT_DIR, 'weights', 'cf.pth')
    focal_weight_path = os.path.join(CURRENT_DIR, 'weights', 'focal.pth')
    map_filepath = os.path.join(CURRENT_DIR, '..', 'map_data', 'rescaled_map.png')
    # map_filepath = os.path.join(CURRENT_DIR, '..', 'map_data', 'map.png')
    start_filepath = os.path.join(CURRENT_DIR, '..', 'map_data', 'start_map.png')
    # start_filepath = os.path.join(CURRENT_DIR, '..', 'map_data', 'start.png')
    goal_filepath = os.path.join(CURRENT_DIR, '..', 'map_data', 'goal_map.png')
    # goal_filepath = os.path.join(CURRENT_DIR, '..', 'map_data', 'goal.png')

    model_cf = Autoencoder(mode='k', resolution=(64, 64))
    model_focal = Autoencoder(mode='f', resolution=(64, 64))

    model_cf.load_state_dict(torch.load(cf_weight_path))
    model_cf.eval()

    model_focal.load_state_dict(torch.load(focal_weight_path))
    model_focal.eval()

    map_design = load_image_tensor(map_filepath)
    start = load_image_tensor(start_filepath)
    goal = load_image_tensor(goal_filepath)

    inputs_g = torch.cat([map_design, goal], dim=1)
    inputs_sg = torch.cat([map_design, start + goal], dim=1)

    f_planner = DifferentiableDiagAstar(mode='f')
    fw100_planner = DifferentiableDiagAstar(mode='f', f_w=100)
    cf_planner = DifferentiableDiagAstar(mode='k')
    w2_planner = DifferentiableDiagAstar(mode='default', h_w=2)
    vanilla_planner = DifferentiableDiagAstar(mode='default', h_w=1)

    with torch.no_grad():
        pred_f = (model_focal(inputs_sg) + 1) / 2
        pred_cf = (model_cf(inputs_g) + 1) / 2

        '''
        outputs_f = f_planner(
            pred_f,
            start,
            goal,
            (map_design == 0) * 1.
        )
        '''

        outputs_cf = cf_planner(
            pred_cf,
            start,
            goal,
            (map_design == 0) * 1.
        )

        outputs_fw100 = fw100_planner(
            pred_f,
            start,
            goal,
            (map_design == 0) * 1.
        )

        outputsw2 = w2_planner(
            (map_design == 0) * 1.,
            start,
            goal,
            (map_design == 0) * 1.
        )

        outputs_vanilla = vanilla_planner(
            (map_design == 0) * 1.,
            start,
            goal,
            (map_design == 0) * 1.
        )

    fig, ax = plt.subplots(1, 6, figsize=(25, 42))

    ax[0].imshow(
        transform_plan(torch.moveaxis(torch.cat(
            [map_design, outputs_vanilla.paths, outputs_vanilla.histories - outputs_vanilla.paths], dim=1)[0], 0, 2)
                       ))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title(f'A* search')

    ax[1].imshow(
        transform_plan(torch.moveaxis(torch.cat(
            [map_design, outputsw2.paths, outputsw2.histories - outputsw2.paths], dim=1)[0], 0, 2)
                       ))
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title(f'WA* search')

    ax[2].imshow(
        transform_plan(torch.moveaxis(torch.cat(
            [map_design, outputs_fw100.paths, outputs_fw100.histories - outputs_fw100.paths], dim=1)[0], 0, 2)
                       ))
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_title(f'Focal search + PPM')

    ax[3].imshow(
        transform_plan(torch.moveaxis(torch.cat(
            [map_design, outputs_cf.paths, outputs_cf.histories - outputs_cf.paths], dim=1)[0], 0, 2)
                       ))
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].set_title(f'WA* search + CF')

    ax[4].imshow(pred_f[0, 0], cmap='gray')
    ax[4].set_xticks([])
    ax[4].set_yticks([])
    ax[4].set_title(f'predicted path probability map')

    ax[5].imshow(pred_cf[0, 0])
    ax[5].set_xticks([])
    ax[5].set_yticks([])
    ax[5].set_title(f'predicted correction factor')

    plt.show()



if __name__ == "__main__":
    infer_path()

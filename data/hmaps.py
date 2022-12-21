import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset


def proc_grid(grid):
    rows = []
    for row in grid:
        rows.append([float(i) for i in row.text.split()])
    return np.array(rows)


class PathData(Dataset):
    def __init__(self, xml_path, koef_path, h_path, grid_size=64, limit_k=1, clip_value=0.):
        self.xml_path = xml_path
        self.koef_path = koef_path
        self.h_path = h_path
        self.file_names = os.listdir(self.xml_path)
        self.size = len(self.file_names) // limit_k
        self.grid_size = grid_size
        self.clip_value = clip_value

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        tree = ET.parse(os.path.join(self.xml_path, self.file_names[idx]))

        root = tree.getroot()

        map_designs = ((torch.tensor(proc_grid(root[0][6])) == 1) * 1.).reshape(1, self.grid_size, self.grid_size)
        hm = torch.FloatTensor(proc_grid(root[1][2])).reshape(1, self.grid_size, self.grid_size)
        if self.clip_value != 0:
            hm = torch.where(hm >= self.clip_value, hm, torch.zeros_like(hm))

        start_maps = torch.zeros_like(map_designs)
        start = (int(root[0][1].text), int(root[0][0].text))
        start_maps[0][start[0], start[1]] = 1

        goal_maps = torch.zeros_like(map_designs)
        goal = (int(root[0][3].text), int(root[0][2].text))
        goal_maps[0][goal[0], goal[1]] = 1

        koef = torch.zeros_like(hm)
        if self.koef_path is not None:
            tree = ET.parse(os.path.join(self.koef_path, self.file_names[idx]))
            root = tree.getroot()
            koef = torch.FloatTensor(proc_grid(root[1][2])).reshape(1, self.grid_size, self.grid_size)

        h = torch.zeros_like(hm)
        if self.h_path is not None:
            tree = ET.parse(os.path.join(self.h_path, self.file_names[idx]))
            root = tree.getroot()
            h = torch.FloatTensor(proc_grid(root[1][2])).reshape(1, self.grid_size, self.grid_size)
        
        return map_designs, start_maps, goal_maps, hm, koef, h

    
class OODMaps(Dataset):
    def __init__(self, xml_path, grid_size=64, clip_value=0.):
        self.xml_path = xml_path
        self.file_names = os.listdir(self.xml_path)
        self.size = len(self.file_names) // 3
        self.grid_size = grid_size
        self.clip_value = clip_value

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        tree = ET.parse(os.path.join(self.xml_path, f'heatmap_{idx}_f.xml'))

        root = tree.getroot()

        map_designs = ((torch.tensor(proc_grid(root[0][6])) == 1) * 1.).reshape(1, self.grid_size, self.grid_size)
        hm = torch.FloatTensor(proc_grid(root[1][2])).reshape(1, self.grid_size, self.grid_size)
        if self.clip_value != 0:
            hm = torch.where(hm >= self.clip_value, hm, torch.zeros_like(hm))

        start_maps = torch.zeros_like(map_designs)
        start = (int(root[0][1].text), int(root[0][0].text))
        start_maps[0][start[0], start[1]] = 1

        goal_maps = torch.zeros_like(map_designs)
        goal = (int(root[0][3].text), int(root[0][2].text))
        goal_maps[0][goal[0], goal[1]] = 1

        koef = torch.zeros_like(hm)
        tree = ET.parse(os.path.join(self.xml_path, f'heatmap_{idx}_k.xml'))
        root = tree.getroot()
        koef = torch.FloatTensor(proc_grid(root[1][2])).reshape(1, self.grid_size, self.grid_size)

        h = torch.zeros_like(hm)
        tree = ET.parse(os.path.join(self.xml_path, f'heatmap_{idx}_h.xml'))
        root = tree.getroot()
        h = torch.FloatTensor(proc_grid(root[1][2])).reshape(1, self.grid_size, self.grid_size)
        
        return map_designs, start_maps, goal_maps, hm, koef, h

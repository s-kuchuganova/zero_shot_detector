from torch.utils.data import Dataset
from glob import glob
import os
import torch
import json
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, root_dir, task, transforms=None, device="cpu"):
        self.task = task
        self.root_dir = root_dir
        self.transforms = transforms
        self.device = device
        self.valid_jsons = self.__get_valid_jsons__()

    def __get_valid_jsons__(self):
        json_paths = [
            elem
            for elem in glob(f"{os.path.join(self.root_dir, self.task)}/*.json")
            if os.stat(elem).st_size > 2
        ]
        return json_paths

    def __len__(self):
        return len(self.valid_jsons)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        json_path = self.valid_jsons[idx]
        data = json.load(open(json_path))
        center = torch.tensor([data[0]["x"], data[0]["y"]])

        img_path = json_path.replace(".json", ".jpg")
        image_source = Image.open(img_path).convert("RGB")
        image_transformed = self.transforms(image_source)
        sample = {
            "image": image_transformed,
            "center": center,
            "promts": (self.task).replace("_", " "),
        }
        return sample

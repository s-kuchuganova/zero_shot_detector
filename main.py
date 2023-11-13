import json
import os
import numpy as np
import datetime
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image


from GroundingDINO.groundingdino.util.inference import load_model
from dataset import CustomDataset
from utils import detect

BOX_TRESHOLD = 0.2
TEXT_TRESHOLD = 0.25

WEIGHTS_PATH = "/app/groundingdino_swint_ogc.pth"
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
TASKS_DIR = "/tasks"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


model = load_model(CONFIG_PATH, WEIGHTS_PATH)

transforms = T.Compose(
    [
        # T.RandomResize([800], max_size=1333),
        T.Resize((320, 320)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

result_dists = []
start_time = datetime.datetime.now()

for task in os.listdir(TASKS_DIR):
    task_dataset = CustomDataset(root_dir=TASKS_DIR, task=task, transforms=transforms)
    dataloader = DataLoader(task_dataset, batch_size=2)
    for batch in tqdm(dataloader):
        centers, logits, dists = detect(
            images=batch["image"],
            text_prompts=batch["promts"],
            real_centers=batch["center"],
            model=model,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=device,
        )
        result_dists.extend(dists)

script_time = datetime.datetime.now() - start_time

result_dists = np.array(result_dists)
print(f"mean accuracy:{sum(result_dists < 0.1) / len(result_dists)}")
print(f"mean distance:{sum(result_dists) / len(result_dists)}")
print(f"script time: {script_time}")

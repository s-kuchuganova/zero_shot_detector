import json
import os
import numpy as np
import datetime
from tqdm import tqdm
from glob import glob
from scipy.spatial import distance
import torch

from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict


BOX_TRESHOLD = 0.2
TEXT_TRESHOLD = 0.25

WEIGHTS_PATH = '/app/groundingdino_swint_ogc.pth'
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
TASKS_DIR = "/tasks"

if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'

model = load_model(CONFIG_PATH, WEIGHTS_PATH)
result = []

start_time = datetime.datetime.now()

for task in os.listdir(TASKS_DIR):
    TEXT_PROMPT = task.replace("_", " ")
    for img in tqdm(glob(f"{os.path.join(TASKS_DIR,task)}/*.jpg")):

        json_path = img.replace(".jpg", ".json")
        if os.stat(json_path).st_size > 2:
            data = json.load(open(json_path))
            gt_x, gt_y = data[0]["x"], data[0]["y"]

            image_source, image = load_image(img)
            w, h = image_source.shape[:2]

            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD,
                device=device, 
            )

            boxes = boxes.tolist()

            try:
                boxes_area = [elem[2] * elem[3] for elem in boxes]
                box_idx = np.argmin(boxes_area)
                x_center, y_center = boxes[box_idx][:2]
                dist = distance.euclidean((gt_x, gt_y), (x_center, y_center))
                result.append(dist)
            except Exception as error:
                print(error)
script_time = datetime.datetime.now() - start_time

result = np.array(result)
print(f'mean accuracy:{sum(result < 0.1) / len(result)}')
print(f'mean distance:{sum(result) / len(result)}')
print(f'script time: {script_time}')

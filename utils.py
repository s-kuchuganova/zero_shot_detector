import torch
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def detect(
    images,
    text_prompts,
    real_centers,
    model,
    box_threshold=0.2,
    text_threshold=0.25,
    device="cuda",
):
    model = model.to(device)

    captions = [preprocess_caption(caption) for caption in text_prompts]

    images = images.to(device)

    with torch.no_grad():
        outputs = model(images, captions=captions)

    prediction_logits = (
        outputs["pred_logits"].cpu().sigmoid()
    )  # prediction_logits.shape = (bsz，nq, 256)
    prediction_boxes = outputs[
        "pred_boxes"
    ].cpu()  # prediction_boxes.shape = (bsz, nq, 4)

    logits_res = []
    centers_res = []
    dists_res = []

    for ub_logits, ub_boxes, gt_center in zip(
        prediction_logits, prediction_boxes, real_centers
    ):
        mask = ub_logits.max(dim=1)[0] > box_threshold
        logits = ub_logits[mask]  # logits.shape = (n, 256)
        boxes = ub_boxes[mask]  # boxes.shape = (n, 4)
        logits_res.append(logits.max(dim=1)[0])

        boxes_area = [elem[2] * elem[3] for elem in boxes.tolist()]
        box_idx = np.argmin(boxes_area)
        x_center, y_center = boxes.tolist()[box_idx][:2]
        gt_x, gt_y = gt_center[0], gt_center[1]
        centers_res.append((x_center, y_center))

        dist = distance.euclidean((gt_x, gt_y), (x_center, y_center))
        dists_res.append(dist)

    return centers_res, logits_res, dists_res


def plot_batch_with_points(
    images, real_points, predicted_points, titles=None, rows=2, cols=4
):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.5)

    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            axes[i, j].imshow(np.transpose(images[index], (1, 2, 0)))
            axes[i, j].axis("off")
            w, h = images[index].shape[1:]

            # Рисуем реальную точку зеленым цветом
            if real_points is not None:
                real_x, real_y = real_points[index]
                axes[i, j].scatter(
                    real_x * w, real_y * h, color="green", marker="o", label="Real"
                )

            # Рисуем предсказанную точку красным цветом
            if predicted_points is not None:
                pred_x, pred_y = predicted_points[index]
                axes[i, j].scatter(
                    pred_x * w, pred_y * h, color="red", marker="x", label="Predicted"
                )

            if titles:
                axes[i, j].set_title(titles[index])

    plt.show()

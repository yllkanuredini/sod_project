import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from data_loader import get_dataloaders
from sod_model import SODModel
from utils import calculate_iou, calculate_precision_recall_f1


def create_overlay(image, mask):
    image = image.copy()
    mask = mask.copy()

    red_overlay = np.zeros_like(image)
    red_overlay[:, :, 0] = mask

    overlay = 0.7 * image + 0.3 * red_overlay
    overlay = np.clip(overlay, 0, 1)

    return overlay


def evaluate_model(model, test_loader, device):
    model.eval()

    total_iou = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_time = 0
    total_batches = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            start_time = time.time()
            predictions = model(images)
            end_time = time.time()

            inference_time = (end_time - start_time) / images.size(0)
            total_time += inference_time

            iou = calculate_iou(predictions, masks)
            precision, recall, f1 = calculate_precision_recall_f1(predictions, masks)

            total_iou += iou.item()
            total_precision += precision.item()
            total_recall += recall.item()
            total_f1 += f1.item()
            total_batches += 1

    results = {
        "IoU": total_iou / total_batches,
        "Precision": total_precision / total_batches,
        "Recall": total_recall / total_batches,
        "F1-score": total_f1 / total_batches,
        "Inference time per image": total_time / total_batches
    }

    return results


def save_visualizations(model, test_loader, device, output_dir, num_samples=5):
    os.makedirs(output_dir, exist_ok=True)

    model.eval()

    images, masks = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        predictions = model(images)

    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    predictions = predictions.cpu().numpy()

    for i in range(min(num_samples, len(images))):
        image = np.transpose(images[i], (1, 2, 0))
        ground_truth = masks[i][0]
        prediction = predictions[i][0]

        binary_prediction = (prediction > 0.5).astype(np.float32)
        overlay = create_overlay(image, binary_prediction)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 4, 1)
        plt.imshow(image)
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(ground_truth, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(binary_prediction, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis("off")

        save_path = os.path.join(output_dir, f"sample_{i + 1}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"Saved visualization: {save_path}")


def main():
    images_dir = "data/ecssd/images"
    masks_dir = "data/ecssd/masks/ground_truth_mask"

    model_path = "outputs/checkpoints/best_sod_model.pth"
    visualization_dir = "outputs/visualizations"

    batch_size = 8
    image_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader = get_dataloaders(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=batch_size,
        image_size=image_size
    )

    model = SODModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print("Best model loaded successfully.")

    results = evaluate_model(model, test_loader, device)

    print("\nTest Set Results")
    print("----------------")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    save_visualizations(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=visualization_dir,
        num_samples=5
    )

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
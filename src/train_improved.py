import os
import torch
import torch.optim as optim
from tqdm import tqdm

from data_loader import get_dataloaders
from sod_model_improved import ImprovedSODModel
from utils import SODLoss, calculate_iou, calculate_precision_recall_f1


def train_one_epoch(model, train_loader, loss_function, optimizer, device):
    model.train()

    total_loss = 0
    total_iou = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    for images, masks in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        predictions = model(images)
        loss = loss_function(predictions, masks)

        loss.backward()
        optimizer.step()

        iou = calculate_iou(predictions, masks)
        precision, recall, f1 = calculate_precision_recall_f1(predictions, masks)

        total_loss += loss.item()
        total_iou += iou.item()
        total_precision += precision.item()
        total_recall += recall.item()
        total_f1 += f1.item()

    num_batches = len(train_loader)

    return {
        "loss": total_loss / num_batches,
        "iou": total_iou / num_batches,
        "precision": total_precision / num_batches,
        "recall": total_recall / num_batches,
        "f1": total_f1 / num_batches
    }


def validate_one_epoch(model, val_loader, loss_function, device):
    model.eval()

    total_loss = 0
    total_iou = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)
            loss = loss_function(predictions, masks)

            iou = calculate_iou(predictions, masks)
            precision, recall, f1 = calculate_precision_recall_f1(predictions, masks)

            total_loss += loss.item()
            total_iou += iou.item()
            total_precision += precision.item()
            total_recall += recall.item()
            total_f1 += f1.item()

    num_batches = len(val_loader)

    return {
        "loss": total_loss / num_batches,
        "iou": total_iou / num_batches,
        "precision": total_precision / num_batches,
        "recall": total_recall / num_batches,
        "f1": total_f1 / num_batches
    }


def save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def main():
    images_dir = "data/ecssd/images"
    masks_dir = "data/ecssd/masks/ground_truth_mask"

    checkpoint_dir = "outputs/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    latest_checkpoint_path = f"{checkpoint_dir}/latest_improved_checkpoint.pth"
    best_model_path = f"{checkpoint_dir}/best_improved_sod_model.pth"

    batch_size = 8
    image_size = 128
    learning_rate = 1e-3
    num_epochs = 20
    patience = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader = get_dataloaders(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=batch_size,
        image_size=image_size
    )

    model = ImprovedSODModel().to(device)
    loss_function = SODLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if os.path.exists(latest_checkpoint_path):
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

        print(f"Resuming improved model training from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device
        )

        val_metrics = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            loss_function=loss_function,
            device=device
        )

        print(
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train IoU: {train_metrics['iou']:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f}"
        )

        print(
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            checkpoint_path=latest_checkpoint_path
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_model_path)
            print("Best improved model saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Improved model training completed.")


if __name__ == "__main__":
    main()
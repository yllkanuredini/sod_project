import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


class SODDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=128, augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.augment = augment

        self.image_files = sorted([
            file for file in os.listdir(images_dir)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        self.mask_files = sorted([
            file for file in os.listdir(masks_dir)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if len(self.image_files) != len(self.mask_files):
            raise ValueError("Number of images and masks is not the same.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.image_files[index])
        mask_path = os.path.join(self.masks_dir, self.mask_files[index])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        mask = (mask > 0.5).astype(np.float32)

        if self.augment:
            if np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

            brightness_factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 1)

        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask


def get_dataloaders(images_dir, masks_dir, batch_size=8, image_size=128):
    full_dataset = SODDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=image_size,
        augment=True
    )

    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_dataset.dataset.augment = False
    test_dataset.dataset.augment = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader
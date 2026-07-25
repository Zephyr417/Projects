import numpy as np
import torch
from torch.utils.data import Dataset

def random_crop(image, label, crop_size=256):
    height, width = image.shape[:2]

    top = np.random.randint(0, height - crop_size + 1)
    left = np.random.randint(0, width - crop_size + 1)

    image = image[top:top + crop_size, left:left + crop_size, :]
    label = label[top:top + crop_size, left:left + crop_size, :]

    return image, label


class TissueNetDataset(Dataset):
    def __init__(self, images, labels, training=False, crop_size=256):
        self.images = images
        self.labels = labels
        self.training = training
        self.crop_size = crop_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.training:
            image, label = random_crop(image, label, self.crop_size)

        # Normalize each channel to [0, 1].
        image = image.astype(np.float32, copy=True)

        low = np.percentile(image, 1, axis=(0, 1), keepdims=True).astype(np.float32)
        high = np.percentile(image, 99, axis=(0, 1), keepdims=True).astype(np.float32)

        scale = np.maximum(high - low, np.float32(1e-6))
        np.subtract(image, low, out=image)
        np.divide(image, scale, out=image)
        np.clip(image, 0, 1, out=image)

        # Instance IDs -> binary masks.
        label = (label > 0).astype(np.float32)

        # HWC -> CHW.
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        label = torch.from_numpy(np.ascontiguousarray(label.transpose(2, 0, 1))).float()

        return image, label
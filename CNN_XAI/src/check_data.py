from pathlib import Path

from torchvision import datasets, transforms


BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DATA_DIR = BASE_DIR / "data" / "brain_mri" / "Training"

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

training_dataset = datasets.ImageFolder(
    root=TRAINING_DATA_DIR,
    transform=transform,
)

print("Number of images:", len(training_dataset))
print("Classes:", training_dataset.classes)
print("Class mapping:", training_dataset.class_to_idx)

image, label = training_dataset[0]

print("Image shape:", image.shape)
print("Label:", label)
print("Class name:", training_dataset.classes[label])

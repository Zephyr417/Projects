import argparse
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def find_training_dir(explicit: Path = None) -> Path:
    if explicit and explicit.exists():
        return explicit
    base = Path(__file__).resolve().parent.parent
    candidates = [
        base / "data" / "brain_mri" / "Training",
        base / "data" / "barin_mri" / "Training",
        Path("data") / "brain_mri" / "Training",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Training data directory not found. Checked: " + ", ".join(str(p) for p in candidates))


def make_dataloader(train_dir: Path, batch_size: int):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    if len(dataset) == 0:
        raise RuntimeError(f"No images found in {train_dir}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader, dataset


def build_model(num_classes: int, device: torch.device):
    model = models.resnet18(pretrained=False)
    # replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("Device:", device)

    train_dir = find_training_dir(Path(args.data_dir) if args.data_dir else None)
    print("Using training dir:", train_dir)

    loader, dataset = make_dataloader(train_dir, args.batch_size)
    num_classes = len(dataset.classes)
    print("Found classes:", dataset.classes)

    model = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    batch_count = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        start = time.time()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
            if args.max_batches and batch_count >= args.max_batches:
                break
        elapsed = time.time() - start
        print(f"Epoch {epoch} done. Avg loss: {epoch_loss:.4f}. Time: {elapsed:.1f}s")
        if args.max_batches and batch_count >= args.max_batches:
            print("Reached max_batches limit; finishing early.")
            break

    out_path = Path("outputs")
    out_path.mkdir(exist_ok=True)
    save_file = out_path / "model_smoke.pth"
    torch.save({"model_state": model.state_dict(), "classes": dataset.classes}, save_file)
    print("Saved model to", save_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="Optional explicit training data dir")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-batches", type=int, default=0, help="If >0, stop after this many batches")
    parser.add_argument("--force-cpu", dest="force_cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

import torch

def dice_score(logits, labels, threshold=0.5, eps=1e-6):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    # Calculate one Dice score per image across both output channels.
    dimensions = (1, 2, 3)
    intersection = (predictions * labels).sum(dim=dimensions)
    denominator = predictions.sum(dim=dimensions) + labels.sum(dim=dimensions)
    return (2 * intersection + eps) / (denominator + eps)


def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_images = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        current_batch_size = images.shape[0]
        total_loss += loss.item() * current_batch_size
        total_dice += dice_score(logits.detach(), labels).sum().item()
        total_images += current_batch_size

    average_loss = total_loss / total_images
    average_dice = total_dice / total_images
    return average_loss, average_dice

def evaluate(dataloader, model, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_images = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            current_batch_size = images.shape[0]
            total_loss += loss.item() * current_batch_size
            total_dice += dice_score(logits, labels).sum().item()
            total_images += current_batch_size

    average_loss = total_loss / total_images
    average_dice = total_dice / total_images
    return average_loss, average_dice
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import os
from copy import deepcopy
from PIL import Image
from torchvision.transforms import Compose, ToTensor, CenterCrop

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
        self.masks = deepcopy(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.images[index])
        mask_path = os.path.join(self.masks_dir, self.masks[index])

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image, mask = self.transform(image), self.transform(mask).squeeze()

        return image, mask.long()

    def __len__(self):
        return len(self.images)


train_dir = "data/reduced/train/"
train_dir_gt = "data/reduced/train_labels/"
val_dir = "data/reduced/val/"
val_dir_gt = "data/reduced/val_labels/"

classes = ('background', 'building',)
labels = ((0, 0, 0), (255, 255, 255),)
model = deeplabv3_resnet101(pretrained=False, num_classes=len(classes)).to(DEVICE)
epochs = 100

criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

transform = Compose([ToTensor(), CenterCrop((256, 256))])

train_dataset = CustomDataset(train_dir, train_dir_gt, transform)
val_dataset = CustomDataset(val_dir, val_dir_gt, transform)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=4, drop_last=True)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=4, drop_last=True)


def compute_iou(preds, targets, num_classes):
    iou_list = []
    preds = preds.argmax(dim=1)  # Convert logits to class predictions
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)

        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()

        if union == 0:
            iou = float('nan')  # Handle division by zero if there are no pixels for the class
        else:
            iou = intersection / union

        iou_list.append(iou)

    return iou_list


best_loss = np.inf
for epoch in range(0, epochs + 1):
    print(f'Epoch {epoch}')
    model.train()
    total_train_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)['out']
        loss = criterion(outputs, targets)
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Train loss: {avg_train_loss}")

    print("Validation:")
    run_val_loss = 0.0
    total_val_iou = np.zeros(len(classes))
    num_batches = 0

    model.eval()
    with torch.no_grad():
        for val_images, val_targets in val_loader:
            val_images, val_targets = val_images.to(DEVICE), val_targets.to(DEVICE)
            val_outputs = model(val_images)['out']
            val_loss = criterion(val_outputs, val_targets)
            run_val_loss += val_loss.item()

            iou_list = compute_iou(val_outputs, val_targets, len(classes))
            total_val_iou += np.array(iou_list)
            num_batches += 1

        avg_val_loss = run_val_loss / num_batches
        avg_val_iou = total_val_iou / num_batches

        print(f"Validation Loss: {avg_val_loss}")
        print(f"Validation IoU: {avg_val_iou}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print("Model saved to best_model.pt")
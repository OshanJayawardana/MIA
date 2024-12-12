from torchvision.models import resnet18
import torch
import torch.nn as nn
from preprocess import MembershipDataset, PseudoLabelDataset
import torchvision.transforms as transforms
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_resnet18(dataset: MembershipDataset, epochs: int, apply_transforms: bool = True):
    """Train a resnet18 model on the dataset. Apply random transformations to the images if apply_transforms is True."""
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 44)
    model.to(device)

    # only fine tune the fc layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    transform = transforms.Compose([
        transforms.RandomResizedCrop((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ])

    imgs, labels = dataset[:][1], dataset[:][4]
    if apply_transforms:
        imgs = [transform(img) for img in imgs]
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels)

    # shuffle data
    indices = torch.randperm(len(imgs))
    imgs = imgs[indices]
    labels = labels[indices]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(imgs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # calculate accuracy
        _, preds = torch.max(outputs, 1)
        correct = (preds == labels.to(device)).sum().item()
        acc = correct / len(labels)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Accuracy: {acc}")

    return model


def load_target_model():
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 44)

    ckpt = torch.load("models/01_MIA_67.pt", map_location=device)

    model.load_state_dict(ckpt)
    return model

if __name__ == "__main__":
    data: MembershipDataset = torch.load("data/pub.pt")
    train_resnet18(data, 1)

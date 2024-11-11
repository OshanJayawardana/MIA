from torchvision.models import resnet18
import torch
import torch.nn as nn
from preprocess import MembershipDataset
import torchvision.transforms as transforms
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_resnet18(dataset: MembershipDataset, epochs: int, apply_transforms: bool = True):
    """Train a resnet18 model on the dataset. Apply random transformations to the images if apply_transforms is True."""
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 44)
    
    # Randomize the weights
    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    model.apply(initialize_weights)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    transform = transforms.Compose([
        transforms.RandomResizedCrop((32, 32)),
        transforms.RandomHorizontalFlip(),
    ])

    imgs, labels = dataset[:][1], dataset[:][2]
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

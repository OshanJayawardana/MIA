from preprocess import PseudoLabelDataset
from process import get_sample
from torchvision.models import resnet18
import torchvision.transforms as transforms
import torch

train_dataset: PseudoLabelDataset = torch.load("data/pub_w_pseudo.pt")
# get index of samples with membership 0
non_members_ids = [
    i for i in range(len(train_dataset)) if train_dataset.membership[i] == 0
]

# get index of samples with membership 1
members_ids = [i for i in range(len(train_dataset)) if train_dataset.membership[i] == 1]
# get the samples with membership 0
non_members = get_sample(train_dataset, non_members_ids)
# get the samples with membership 1
members = get_sample(train_dataset, members_ids)

val_dataset: PseudoLabelDataset = torch.load("data/priv_out_w_pseudo.pt")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
apply_transforms = False

def train_resnet18(
    dataset: PseudoLabelDataset, val_dataset: PseudoLabelDataset, epochs: int
):
    """Train a resnet18 model on the dataset. Apply random transformations to the images if apply_transforms is True."""
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 44)
    model.to(device)

    # only fine tune the fc layer
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.fc.parameters():
    #     param.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

    transform = transforms.Compose(
        [
            transforms.RandAugment(),
        ]
    )

    imgs, labels = dataset[:][1], dataset[:][2]
    val_imgs, val_labels = val_dataset[:][1], val_dataset[:][2]

    if apply_transforms:
        imgs = [transform(img) for img in imgs]
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels)

    val_imgs = torch.stack(val_imgs)[:1000]
    val_labels = torch.tensor(val_labels)[:1000]

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

        # calculate accuracy on validation set
        val_outputs = model(val_imgs.to(device))
        val_loss = criterion(val_outputs, val_labels.to(device))

        _, val_preds = torch.max(val_outputs, 1)
        val_correct = (val_preds == val_labels.to(device)).sum().item()
        val_acc = val_correct / len(val_labels)

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Accuracy: {acc}, Val Loss: {val_loss.item()}, Val Accuracy: {val_acc}"
        )

    return model


for num in range(4):
    if num % 2 == 0:
        model = train_resnet18(non_members, val_dataset, 10)
    else:
        model = train_resnet18(members, val_dataset, 10)
    torch.save(model, f"models/shadow_{num}.pt")

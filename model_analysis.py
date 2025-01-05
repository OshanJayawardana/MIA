from model import load_target_model
from preprocess import MembershipDataset, PseudoLabelDataset
from process import append_sample, pop_sample, get_sample
from infer import infer
import torch

model = load_target_model()
model.eval()

train_dataset : PseudoLabelDataset = torch.load("data/pub_w_pseudo.pt")
membership = torch.tensor(train_dataset[:][3])

# select indices of the samples that are in the membership
indices = torch.where(membership == 1)[0]
print("Number of samples in the membership:", len(indices))
dataset_mem = get_sample(train_dataset, indices)

# select indices of the samples that are not in the membership
indices = torch.where(membership == 0)[0]
print("Number of samples not in the membership:", len(indices))
dataset_non_mem = get_sample(train_dataset, indices)

def add_gaussian_noise(dataset, mean=0, std=0.01):
    """Add Gaussian noise to the dataset.
    """
    new_dataset = PseudoLabelDataset()
    new_dataset.ids = dataset.ids
    new_dataset.imgs = [img + torch.randn_like(img) * std + mean for img in dataset.imgs]
    new_dataset.labels = dataset.labels
    new_dataset.membership = dataset.membership
    new_dataset.pseudo_labels = dataset.pseudo_labels
    return new_dataset

dataset_mem = add_gaussian_noise(dataset_mem)
dataset_non_mem = add_gaussian_noise(dataset_non_mem)

def calculate_accuracy(model, dataset):
    labels = torch.tensor(dataset[:][2])
    logits = infer(model, dataset)

    # calculate the accuracy of the model
    correct = 0
    for i in range(len(labels)):
        if torch.argmax(logits[i]) == labels[i]:
            correct += 1

    accuracy = correct / len(labels)
    return accuracy

accuracy_mem = calculate_accuracy(model, dataset_mem)
print(f"Accuracy Member: {accuracy_mem}")

accuracy_non_mem = calculate_accuracy(model, dataset_non_mem)
print(f"Accuracy Non-Member: {accuracy_non_mem}")
from preprocess import MembershipDataset
from model import load_target_model
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_model = load_target_model()
target_model.eval()
target_model.to(device)

def add_random_patch(image, patch_size_range=(8, 16)):
    """
    Adds a random non-symmetric patch to a given PyTorch image.

    Parameters:
        image (torch.Tensor): Input image of shape (3, 32, 32).
        patch_size_range (tuple): Min and max size for the random patch width and height.

    Returns:
        torch.Tensor: Image with the random patch added.
    """
    # Ensure the image has the correct shape (C, H, W)
    assert image.shape == (3, 32, 32), "Image must be of shape (3, 32, 32)"
    
    # Generate random patch width and height
    patch_width = 16
    patch_height = 16
    
    # Generate random coordinates for the top-left corner of the patch
    max_x = 32 - patch_width
    max_y = 32 - patch_height
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    
    # Create a random patch with the generated width and height
    patch = torch.rand((3, patch_height, patch_width), device=image.device)
    
    # Apply the patch to the image
    patched_image = image.clone()
    patched_image[:, y:y+patch_height, x:x+patch_width] = patch
    
    return patched_image

def apply_random_patches(img: torch.Tensor, n: int=10) -> torch.Tensor:
    """
    Apply n random patches to the image and return n images.
    """
    return torch.stack([add_random_patch(img) for _ in range(n)])

def get_logit_distributions(dataset: MembershipDataset, model: torch.nn.Module) -> torch.Tensor:
    """
    For each sample in the dataset generate n images with random patches and return the logits(for original label)
    of the model for each image.
    """
    imgs = torch.stack(dataset.imgs).to(device)
    labels = torch.tensor(dataset.labels).to(device)
    membership = dataset.membership

    # calculate original logits
    original_outputs = model(imgs)
    original_logits = original_outputs[range(len(labels)), labels]

    logit_distributions = []

    for img, label, original_logit in zip(imgs, labels, original_logits):
        label = label.unsqueeze(0)

        # Generate n images with random patches
        n = 10
        perturbed_imgs = apply_random_patches(img.squeeze(), n)
        labels_repeated = torch.repeat_interleave(label, n, dim=0)

        # Forward pass for each image
        perturbed_outputs = model(perturbed_imgs.to(device))
        selected_logits = perturbed_outputs[range(n), labels_repeated]

        # Calculate the difference between the original logit and the logit of the perturbed image
        selected_logits = torch.mean(original_logit - selected_logits)

        logit_distributions.append(selected_logits.detach().cpu())

    return torch.tensor(logit_distributions), membership

# dataset = torch.load("data/pub.pt")
# logit_distributions, new_membership = get_logit_distributions(dataset, target_model)

# # Plot two logit histograms for membership 0 and 1 in the same plot
# non_members = [logit.item() for i, logit in enumerate(logit_distributions) if new_membership[i] == 0]
# members = [logit.item() for i, logit in enumerate(logit_distributions) if new_membership[i] == 1]

# fig, axs = plt.subplots(2, 1, sharex=True)
# axs[0].hist(non_members, bins=50, color="red")
# axs[0].set_title("Non-members")
# axs[0].set_xlabel("Logit")
# axs[0].set_ylabel("Frequency")
# axs[1].hist(members, bins=50, color="blue")
# axs[1].set_title("Members")
# axs[1].set_xlabel("Logit")
# axs[1].set_ylabel("Frequency")
# plt.tight_layout()
# plt.savefig("logit_histograms.pdf")

test_dataset = torch.load("data/priv_out.pt")
logit_distributions, new_membership = get_logit_distributions(test_dataset, target_model)
ids = test_dataset.ids
score = logit_distributions
df = pd.DataFrame({"ids": ids, "score": score})
df.to_csv("results/patch.csv", index=False)
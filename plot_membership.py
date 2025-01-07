from preprocess import MembershipDataset
import torch
import matplotlib.pyplot as plt
import random

dataset: MembershipDataset = torch.load("data/pub.pt")
non_members = [img.permute(1, 2, 0).numpy() for i, img in enumerate(dataset.imgs) if dataset.membership[i] == 0]
members = [img.permute(1, 2, 0).numpy() for i, img in enumerate(dataset.imgs) if dataset.membership[i] == 1]

# shuffle non-members and members
random.shuffle(non_members)
random.shuffle(members)

# plot first 20 non-members
fig, axs = plt.subplots(4, 5)
for i, ax in enumerate(axs.flat):
    ax.imshow(non_members[i])
    ax.axis("off")
plt.suptitle("Non-members")
plt.savefig("figures/non_members.png")

# plot first 20 members
fig, axs = plt.subplots(4, 5)
for i, ax in enumerate(axs.flat):
    ax.imshow(members[i])
    ax.axis("off")
plt.suptitle("Members")
plt.savefig("figures/members.png")

# from each class(0-43) plot 5 random images
label_points = {i: [img.permute(1, 2, 0).numpy() for j, img in enumerate(dataset.imgs) if dataset.labels[j] == i] for i in range(44)}
fig, axs = plt.subplots(44, 4, figsize=(1.28, 14.08))
for i in range(44):
    random.shuffle(label_points[i])
    for j, ax in enumerate(axs[i]):
        ax.imshow(label_points[i][j])
        ax.axis("off")
plt.savefig("figures/class_images.png")



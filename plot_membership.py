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
plt.savefig("non_members.pdf")

# plot first 20 members
fig, axs = plt.subplots(4, 5)
for i, ax in enumerate(axs.flat):
    ax.imshow(members[i])
    ax.axis("off")
plt.suptitle("Members")
plt.savefig("members.pdf")



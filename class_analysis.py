from preprocess import MembershipDataset
import torch
import matplotlib.pyplot as plt
import numpy as np
import random

dataset: MembershipDataset = torch.load("data/pub.pt")

# for members and non-members plot histograms of class labels
non_members = [label for i, label in enumerate(dataset.labels) if dataset.membership[i] == 0]
members = [label for i, label in enumerate(dataset.labels) if dataset.membership[i] == 1]

# print the class label that has lowest frequency for members
print(min(set(members), key=members.count))

fig, axs = plt.subplots(2, 1)
axs[0].hist(non_members, bins=np.arange(0, 44, 1), color="red")
axs[0].set_title("Non-members")
axs[0].set_xlabel("Class label")
axs[0].set_ylabel("Frequency")
axs[1].hist(members, bins=np.arange(0, 44, 1), color="blue")
axs[1].set_title("Members")
axs[1].set_xlabel("Class label")
axs[1].set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("figures/class_histograms.png")
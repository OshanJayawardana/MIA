from preprocess import MembershipDataset
from sklearn.decomposition import PCA
import torch
import numpy as np
import matplotlib.pyplot as plt

dataset = torch.load("data/pub.pt")
imgs = torch.stack(dataset.imgs).view(len(dataset.imgs), -1).numpy()
label = np.array(dataset.labels)

pca = PCA(n_components=3)
pca.fit(imgs)
transformed = pca.transform(imgs)
label_points = {i: transformed[label == i] for i in range(44)}
# plot pca and color label by class, has 44 classes total
for i in range(44):
    points = label_points[i]
    plt.scatter(points[:, 0], points[:, 1], label=f"Class {i}")
plt.legend()
plt.savefig("pca.pdf")

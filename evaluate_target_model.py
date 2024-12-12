from model import load_target_model
from preprocess import MembershipDataset
from infer import infer
import torch
from sklearn.metrics import accuracy_score
import numpy as np

model = load_target_model()
data: MembershipDataset = torch.load("data/pub.pt")
ground_truth = np.array(data.labels)
scores = infer(model, data).argmax(dim=1).numpy()
acc = accuracy_score(ground_truth, scores)
print(ground_truth[:10])
print(scores[:10])
print(f"Accuracy on public data: {acc}")
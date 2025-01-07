import pickle
from model import load_target_model
from preprocess import ActivationDataset, GradientDataset
import torch
from sklearn import tree, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from infer import infer
import pandas as pd
import numpy as np

def get_sample(dataset: ActivationDataset, indexes: list) -> ActivationDataset:
    new_dataset = ActivationDataset()
    new_dataset.ids = [dataset.ids[i] for i in indexes]
    new_dataset.imgs = [dataset.imgs[i] for i in indexes]
    new_dataset.labels = [dataset.labels[i] for i in indexes]
    new_dataset.membership = [dataset.membership[i] for i in indexes]
    new_dataset.activations = [dataset.activations[i] for i in indexes]
    return new_dataset

model = load_target_model()
model.eval()
dataset : ActivationDataset = torch.load("data/pub_w_activations.pt")

model_list = []

for j in range(44):
    indexes = [i for i, label in enumerate(dataset.labels) if label == j]
    sub_dataset = get_sample(dataset, indexes)
    y = np.array(sub_dataset.membership)
    X = torch.stack(sub_dataset.activations).cpu().numpy()

    classifier = svm.SVC()
    classifier.fit(X, y)

    model_list.append(classifier)

test_dataset : ActivationDataset = torch.load("data/priv_out_w_activations.pt")
ids = test_dataset.ids
score = []
for sample in test_dataset:
    activations = sample[4].cpu().numpy().reshape(1, -1)
    selected_model = model_list[sample[2]]
    prediction = selected_model.decision_function(activations)
    score.append(prediction.item())

# save ids and score to a pandas dataframe
df = pd.DataFrame({"ids": ids, "score": score})
df.to_csv("results/activation.csv", index=False)
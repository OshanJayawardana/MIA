import pickle
from model import load_target_model
from preprocess import ActivationDataset, GradientDataset, LogitstDataset
import torch
from sklearn import tree, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from infer import infer
import pandas as pd

def get_sample(dataset: LogitstDataset, indexes: list) -> LogitstDataset:
    new_dataset = LogitstDataset()
    new_dataset.ids = [dataset.ids[i] for i in indexes]
    new_dataset.imgs = [dataset.imgs[i] for i in indexes]
    new_dataset.labels = [dataset.labels[i] for i in indexes]
    new_dataset.membership = [dataset.membership[i] for i in indexes]
    new_dataset.logits = dataset.logits[indexes]
    return new_dataset

model = load_target_model()
model.eval()
dataset : LogitstDataset = torch.load("data/pub_w_pool.pt")

model_list = []

for j in range(44):
    try:
        indexes = [i for i, label in enumerate(dataset.labels) if label == j]
        sub_dataset = get_sample(dataset, indexes)
        y = torch.tensor(sub_dataset[:][3])
        y = y.cpu().numpy()
        X = sub_dataset[:][4]
        # X = X[:, 2].reshape(-1, 1)

        classifier = svm.SVC()
        classifier.fit(X, y)

        model_list.append(classifier)

    except Exception as e:
        continue

test_dataset : LogitstDataset = torch.load("data/priv_out_w_pool.pt")
ids = test_dataset.ids
score = []
for sample in test_dataset:
    logits = sample[4].reshape(1, -1)
    selected_model = model_list[sample[2]]
    prediction = selected_model.decision_function(logits)
    score.append(prediction.item())

# save ids and score to a pandas dataframe
df = pd.DataFrame({"ids": ids, "score": score})
df.to_csv("results/pool.csv", index=False)
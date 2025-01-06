from preprocess import LogitstDataset
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import torch
import numpy as np
import pandas as pd

def get_sample(dataset: LogitstDataset, indices: list[int]) -> LogitstDataset:
    """Get a sample of the dataset based on the indices."""
    sample = LogitstDataset()
    sample.ids = [dataset.ids[i] for i in indices]
    sample.imgs = [dataset.imgs[i] for i in indices]
    sample.labels = [dataset.labels[i] for i in indices]
    sample.membership = [dataset.membership[i] for i in indices]
    sample.logits = [dataset.logits[i] for i in indices]
    return sample

def cluster_dataset(dataset: LogitstDataset, n_clusters: int) -> tuple[list[LogitstDataset], np.ndarray]:
    """Cluster the public dataset into n_clusters clusters according to imgs"""
    imgs = torch.stack(dataset.imgs).view(len(dataset), -1).numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(imgs)
    clusters = [[] for _ in range(n_clusters)]
    cluster_centers = kmeans.cluster_centers_
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(i)
    return [get_sample(dataset, cluster) for cluster in clusters], cluster_centers

def get_logit_distributions(datasets: list[LogitstDataset]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Get the distribution of logits for each dataset and their membership"""
    distributions = []
    for dataset in datasets:
        logits = np.array(dataset.logits)
        membership = np.array(dataset.membership)
        distributions.append((logits, membership))
    return distributions

def train_svms(distributions: list[tuple[np.ndarray, np.ndarray]]) -> list:
    """Train an SVM for each distribution of logits and membership"""
    svms = []
    for logits, membership in distributions:
        model = SVC(probability=True)
        model.fit(logits, membership)
        model = CalibratedClassifierCV(model)
        model.fit(logits, membership)
        svms.append(model)
    return svms

dataset = torch.load("data/pub_w_logits.pt")
n_clusters = 28
datasets, cluster_centers = cluster_dataset(dataset, n_clusters)
distributions = get_logit_distributions(datasets)
svms = train_svms(distributions)

def get_closest_svm_output(sample: torch.Tensor, logit: np.ndarray, svms: list, cluster_centers: np.ndarray) -> int:
    """Get the index of the closest SVM to the sample"""
    sample = sample.view(1, -1).numpy()
    logit = logit.reshape(1, -1)
    closest_cluster = np.argmin(np.linalg.norm(sample - cluster_centers, axis=1))
    return svms[closest_cluster].predict_proba(logit)[0][1]

test_dataset = torch.load("data/priv_out_w_logits.pt")
ids = test_dataset.ids
imgs = test_dataset.imgs
logits = test_dataset.logits

score = []
for i, sample in enumerate(zip(imgs, logits)):
    img, logit = sample
    prediction = get_closest_svm_output(img, logit, svms, cluster_centers)
    score.append(prediction)

# save ids and score to a pandas dataframe
df = pd.DataFrame({"ids": ids, "score": score})
df.to_csv("results/svm.csv", index=False)


    
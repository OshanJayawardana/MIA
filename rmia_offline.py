import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.cluster import KMeans
import torch
from tqdm import tqdm
import hashlib
from preprocess import PseudoLabelDataset
from process import get_sample
from model import load_target_model, train_resnet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def select_representative_embeddings(
    embeddings, num_representatives=250, random_state=42
):
    """
    Select a representative subset of embeddings to cover the embedding space.

    Parameters:
    - embeddings (numpy.ndarray): Array of embeddings of shape (N, 3, 32, 32).
    - num_representatives (int): The number of representative embeddings to select.
    - random_state (int): Random state for reproducibility.

    Returns:
    - numpy.ndarray: Array of selected representative embeddings.
    """
    # Step 1: Flatten each embedding
    flattened_embeddings = embeddings.reshape(
        embeddings.shape[0], -1
    )  # Shape becomes (N, 3072)

    # Step 2: Apply K-means clustering
    kmeans = KMeans(n_clusters=num_representatives, random_state=random_state)
    kmeans.fit(flattened_embeddings)

    # Step 3: Find the closest embedding to each cluster center
    representative_indices = []
    for i in range(num_representatives):
        # Get all points in the current cluster
        cluster_points = flattened_embeddings[kmeans.labels_ == i]
        cluster_indices = np.where(kmeans.labels_ == i)[0]

        # Find the point closest to the cluster center
        center = kmeans.cluster_centers_[i]
        closest_index = cluster_indices[
            np.argmin(np.linalg.norm(cluster_points - center, axis=1))
        ]
        representative_indices.append(closest_index)

    # Step 4: Return the selected embeddings
    return representative_indices


def sample_datasets(
    source_dataset: PseudoLabelDataset, n: int = 2, random_state: int = 42
) -> list[PseudoLabelDataset]:
    """Devide the dataset into n subsets. The subsets are created by shuffling the dataset and splitting it into n parts."""
    np.random.seed(random_state)
    indexes = np.random.permutation(len(source_dataset))
    # split indexes into n parts
    split = np.array_split(indexes, n)
    datasets = [get_sample(source_dataset, split[i]) for i in range(n)]
    return datasets


def create_dataset_hash(dataset: PseudoLabelDataset) -> str:
    """Create a stable hash of the dataset using SHA-256. The hash is created by concatenating the ids of the dataset
    and hashing the resulting string, ensuring consistency across runs."""
    ids_str = "".join(map(str, dataset.ids)).encode("utf-8")
    return hashlib.sha256(ids_str).hexdigest()[:8]


def create_model_hash(model: torch.nn.Module) -> str:
    """Create a hash of the model. The hash is created by hashing the model state dictionary using SHA-256."""
    model_str = str(model.state_dict()).encode("utf-8")
    return hashlib.sha256(model_str).hexdigest()[:8]


def create_reference_models(
    source_dataset: PseudoLabelDataset, k: int = 8, path: str = "models"
):
    """Create n models and train them on the dataset. The models are trained using the train_resnet18 function
    with variable epochs. The models are stored in a list and returned.
    """
    k = k // 2
    epoch_list = [20] * k
    datasets = sample_datasets(source_dataset, n=2)
    for epoch in epoch_list:
        for i in range(2):
            dataset = datasets[i]
            dataset_hash = create_dataset_hash(dataset)
            model = train_resnet18(dataset, epoch, apply_transforms=False)
            model_hash = create_model_hash(model)
            torch.save(model, f"{path}/reference_{dataset_hash}_{model_hash}.pt")


def pr_x_given_theta(sample: PseudoLabelDataset, model: torch.nn.Module) -> float:
    """Calculate the probability of the sample dataset given the model. This is done by
    infering the given sample, taking the logits, converting them to probabilities using softmax
    and taking the probability of label class
    """
    model.eval()
    model.to(device)
    _, img, label, _, _ = sample[0]
    with torch.no_grad():
        logits = model(img[None, ...].to(device))
        probs = torch.nn.functional.softmax(logits, dim=1)
        prob = probs[0, label]
    return prob.item()


def pr_x_out(sample: PseudoLabelDataset, out_models: list[torch.nn.Module]) -> float:
    probs = [pr_x_given_theta(sample, model) for model in out_models]
    return np.mean(probs)


def pr_x(
    sample: PseudoLabelDataset, out_models: list[torch.nn.Module], a: float
) -> float:
    return 0.5 * ((1 + a) * pr_x_out(sample, out_models) + (1 - a))


def ratio_x(
    sample: PseudoLabelDataset,
    out_models: list[torch.nn.Module],
    target_model: torch.nn.Module,
    a: float,
) -> float:
    return pr_x_given_theta(sample, target_model) / pr_x(sample, out_models, a)


def pr_z_batch(
    z: PseudoLabelDataset, out_models: list[torch.nn.Module]
) -> torch.Tensor:
    """Calculate the probabilities of the distribution dataset given the models in parallel."""
    probs = []
    for model in out_models:
        model.eval()
        model.to(device)
        imgs = torch.stack([img for _, img, _, _, _ in z]).to(device)
        labels = torch.tensor([label for _, _, label, _, _ in z]).to(device)
        with torch.no_grad():
            logits = model(imgs)
            model_probs = torch.nn.functional.softmax(logits, dim=1)
            probs.append(model_probs[range(len(labels)), labels])
    return torch.mean(torch.stack(probs), dim=0)


def ratio_z_batch(
    z: PseudoLabelDataset,
    out_models: list[torch.nn.Module],
    target_model: torch.nn.Module,
) -> torch.Tensor:
    """Calculate the ratio of the probabilities of the distribution dataset given the target model and out models in parallel."""
    target_model.eval()
    target_model.to(device)
    imgs = torch.stack([img for _, img, _, _, _ in z]).to(device)
    labels = torch.tensor([label for _, _, label, _, _ in z]).to(device)
    with torch.no_grad():
        logits = target_model(imgs)
        target_probs = torch.nn.functional.softmax(logits, dim=1)
        target_probs = target_probs[range(len(labels)), labels]
    pr_z_vals = pr_z_batch(z, out_models)
    return target_probs / pr_z_vals


def rmia_score(
    sample: PseudoLabelDataset,
    out_models: list[torch.nn.Module],
    target_model: torch.nn.Module,
    a: float,
    z: PseudoLabelDataset,
    gamma: float = 2,
) -> float:
    count = 0
    ratio_x_val = ratio_x(sample, out_models, target_model, a)
    ratio_z_vals = ratio_z_batch(z, out_models, target_model)
    count = torch.sum(ratio_x_val / ratio_z_vals > gamma).item()
    return count / len(z)


def get_rmia_score(
    test_dataset: PseudoLabelDataset,
    out_models: list[torch.nn.Module],
    target_model: torch.nn.Module,
    a: float,
    z: PseudoLabelDataset,
    gamma: float = 2,
) -> float:
    """Return scores for the test dataset. The score is calculated using the RMIA formula."""
    scores = []
    for i in tqdm(range(len(test_dataset)), desc="Calculating RMIA score"):
        sample = get_sample(test_dataset, i)
        score = rmia_score(sample, out_models, target_model, a, z, gamma)
        scores.append(score)
    return scores


def select_closest_embeddings(
    distribution_data: PseudoLabelDataset, test_dataset: PseudoLabelDataset, n: int
) -> list[int]:
    """From the distribution data, select the n closest embeddings to the test dataset embedding distribution.
    First for each sample in distribution data, calculate the average distance to all samples in the test dataset.
    Then select the n samples with the lowest average distance.
    """
    test_embeddings = np.array([img for _, img, _, _, _ in test_dataset])
    distribution_embeddings = np.array([img for _, img, _, _, _ in distribution_data])
    avg_distances = []
    for i in range(len(distribution_data)):
        avg_distance = np.mean(
            np.linalg.norm(test_embeddings - distribution_embeddings[i], axis=1)
        )
        avg_distances.append(avg_distance)
    indexes = np.argsort(avg_distances)[:n]
    return indexes


def rmia_attack(
    distribution_data: PseudoLabelDataset,
    test_dataset: PseudoLabelDataset,
    k: int,
    a: float,
    num_z: int,
    gamma: float = 2,
    path: str = "models",
    force_model_create: bool = False,
) -> list[float]:
    """Perform the RMIA attack. The attack is performed by splitting the distribution dataset into k parts, training
    k models on each part, and calculating the RMIA score for the test dataset. The RMIA score is calculated using the
    RMIA formula.
    """
    model_names = [name for name in os.listdir(path) if name.endswith(".pt")]
    if len(model_names) == 0 or force_model_create:
        # create k models and store them
        create_reference_models(distribution_data, k=k, path=path)
        model_names = [name for name in os.listdir(path) if name.endswith(".pt")]

    # load the models
    out_models = [torch.load(f"{path}/{name}") for name in model_names]
    # load the target model
    target_model = load_target_model()
    # only pick num_z random samples from the distribution data. if num_z is -1, use all samples
    if num_z != -1:
        # indexes = select_closest_embeddings(distribution_data, test_dataset, num_z)
        indexes = np.random.choice(len(distribution_data), num_z, replace=False)
        z = get_sample(distribution_data, indexes)

    # calculate the RMIA score
    scores = get_rmia_score(test_dataset, out_models, target_model, a, z, gamma)
    return scores, out_models


def cross_val(
    distribution_data: PseudoLabelDataset,
    k: int,
    a: float,
    num_z: int,
    gamma: float = 2,
    num_runs: int = 5,
):
    """Perform cross validation for the RMIA attack. The attack is performed num_runs times and the average score"""
    tpr_at_fpr_list = []
    for _ in range(num_runs):
        # shuffle the distribution data
        indexes = np.random.permutation(len(distribution_data))
        distribution_data = get_sample(distribution_data, indexes)

        # split the dataset in 0.8:0.2 ratio
        split = int(0.8 * len(distribution_data))
        test_dataset = get_sample(
            distribution_data, list(range(split, len(distribution_data)))
        )
        distribution_data = get_sample(distribution_data, list(range(split)))

        with tempfile.TemporaryDirectory() as temp_dir:
            predicted_scores, out_models = rmia_attack(
                distribution_data, test_dataset, k, a, num_z, gamma, path=temp_dir
            )

        predicted_scores = np.array(predicted_scores)
        ground_truth = np.array(test_dataset.membership)

        # calculate area under the curve
        fpr, tpr, _ = roc_curve(ground_truth, predicted_scores)
        # get tpr at fpr = 0.05
        tpr_at_fpr = np.interp(0.05, fpr, tpr)
        print(f"TPR at FPR = 0.05: {tpr_at_fpr}")
        tpr_at_fpr_list.append(tpr_at_fpr)

        if tpr_at_fpr > 0.06:
            print("Saving models")
            # convert the tpr_at_fpr to a string with 4 decimal places
            tpr_at_fpr_str = "{:.4f}".format(tpr_at_fpr)
            # create a directory to store the models
            os.makedirs(f"models/{tpr_at_fpr_str}", exist_ok=True)
            # save the models
            for i, model in enumerate(out_models):
                torch.save(model, f"models/{tpr_at_fpr_str}/model_{i}.pt")

    tpr_at_fpr = np.array(tpr_at_fpr_list)
    return np.mean(tpr_at_fpr), np.std(tpr_at_fpr)


def metrics(predicted_scores: np.array, ground_truth: np.array) -> float:
    fpr, tpr, _ = roc_curve(ground_truth, predicted_scores)
    tpr_at_fpr = np.interp(0.05, fpr, tpr)
    return tpr_at_fpr


if __name__ == "__main__":
    k = 4
    num_z = 250
    a = 0.5
    gamma = 2
    num_runs = 100

    distribution_data: PseudoLabelDataset = torch.load("data/pub_overwrite.pt")
    test_dataset: PseudoLabelDataset = torch.load("data/priv_overwrite.pt")

    # cross validation
    # tpr_at_fpr, std = cross_val(distribution_data, k, a, num_z, gamma, num_runs)
    # print(f"TPR at FPR = 0.05: {tpr_at_fpr} +/- {std}")

    # validation
    # scores, _ = rmia_attack(train_dataset, val_dataset, k, a, num_z, gamma, path="models/val", force_model_create=False)
    # ground_truth = np.array(val_dataset.membership)
    # predicted_scores = np.array(scores)
    # tpr_at_fpr = metrics(scores, ground_truth)
    # print(f"TPR at FPR = 0.05: {tpr_at_fpr}")

    scores, _ = rmia_attack(
        distribution_data,
        test_dataset,
        k,
        a,
        num_z,
        gamma,
        path="models/shadow",
        force_model_create=False,
    )
    df = pd.DataFrame({"ids": test_dataset.ids, "score": scores})
    df.to_csv("rmia_offline_35.csv", index=False)

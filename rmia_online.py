import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.cluster import KMeans
import torch
from tqdm import tqdm
import hashlib
from preprocess import MembershipDataset
from model import load_target_model, train_resnet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def append_sample(dataset: MembershipDataset, sample: MembershipDataset) -> MembershipDataset:
    """Append the sample dataset to the dataset. This is done by concatenating the ids, imgs, labels, and membership
    of the sample dataset to the dataset.
    """
    new_dataset = MembershipDataset()
    new_dataset.ids = dataset.ids + sample.ids
    new_dataset.imgs = dataset.imgs + sample.imgs
    new_dataset.labels = dataset.labels + sample.labels
    new_dataset.membership = dataset.membership + sample.membership
    return new_dataset

def pop_sample(dataset: MembershipDataset, index: int) -> tuple[MembershipDataset, MembershipDataset]:
    """Pop the sample dataset at the index. Return the popped sample and the new dataset.
    """
    sample = get_sample(dataset, index)
    new_dataset = MembershipDataset()
    new_dataset.ids = dataset.ids[:index] + dataset.ids[index+1:]
    new_dataset.imgs = dataset.imgs[:index] + dataset.imgs[index+1:]
    new_dataset.labels = dataset.labels[:index] + dataset.labels[index+1:]
    new_dataset.membership = dataset.membership[:index] + dataset.membership[index+1:]
    return sample, new_dataset

def get_sample(dataset: MembershipDataset, index: int | list[int]) -> MembershipDataset:
    """Get the sample dataset at the index. This is done by creating a new MembershipDataset and copying the ids, imgs,
    labels, and membership of the dataset at the index to the new dataset.
    """
    sample = MembershipDataset()
    if isinstance(index, int):
        index = [index]
    sample.ids = [dataset.ids[i] for i in index]
    sample.imgs = [dataset.imgs[i] for i in index]
    sample.labels = [dataset.labels[i] for i in index]
    sample.membership = [dataset.membership[i] for i in index]
    return sample

def select_representative_embeddings(embeddings, num_representatives=250, random_state=42):
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
    flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1)  # Shape becomes (N, 3072)

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
        closest_index = cluster_indices[np.argmin(np.linalg.norm(cluster_points - center, axis=1))]
        representative_indices.append(closest_index)

    # Step 4: Return the selected embeddings
    return representative_indices

def sample_datasets(source_dataset: MembershipDataset, n: int=2, random_state: int=42) -> list[MembershipDataset]:
    """Devide the dataset into n subsets. The subsets are created by shuffling the dataset and splitting it into n parts.
    """
    np.random.seed(random_state)
    indexes = np.random.permutation(len(source_dataset))
    # split indexes into n parts
    split = np.array_split(indexes, n)
    datasets = [get_sample(source_dataset, split[i]) for i in range(n)]
    return datasets

def create_dataset_hash(dataset: MembershipDataset) -> str:
    """Create a stable hash of the dataset using SHA-256. The hash is created by concatenating the ids of the dataset 
    and hashing the resulting string, ensuring consistency across runs."""
    ids_str = "".join(map(str, dataset.ids)).encode("utf-8")
    return hashlib.sha256(ids_str).hexdigest()[:8]

def create_model_hash(model: torch.nn.Module) -> str:
    """Create a hash of the model. The hash is created by hashing the model state dictionary using SHA-256.
    """
    model_str = str(model.state_dict()).encode('utf-8')
    return hashlib.sha256(model_str).hexdigest()[:8]

def create_online_models(sample: MembershipDataset, distribution_data: MembershipDataset) -> list[torch.nn.Module]:
    """Create n models and train them on the dataset. The models are trained using the train_resnet18 function
    with variable epochs. The models are stored in a list and returned.
    """
    # pick a random 1 index from the distribution data
    index = np.random.randint(len(distribution_data))

    # remove the sample from the distribution data
    _, temp_dataset = pop_sample(distribution_data, index)

    # x_in_dataset is temp_dataset with sample appended
    x_in_dataset = append_sample(temp_dataset, sample)

    # x_out_dataset is distribution data
    x_out_dataset = distribution_data

    models = [train_resnet18(x_in_dataset, 100, apply_transforms=False), train_resnet18(x_out_dataset, 100, apply_transforms=False)]
    return models

def pr_x_given_theta(sample: MembershipDataset, model: torch.nn.Module) -> float:
    """Calculate the probability of the sample dataset given the model. This is done by
    infering the given sample, taking the logits, converting them to probabilities using softmax
    and taking the probability of label class
    """
    model.eval()
    model.to(device)
    _, img, label, _ = sample[0]
    with torch.no_grad():
        logits = model(img[None, ...].to(device))
        probs = torch.nn.functional.softmax(logits, dim=1)
        prob = probs[0, label]
    return prob.item()

def pr_x_out(sample: MembershipDataset, out_models: list[torch.nn.Module]) -> float:
    probs = [pr_x_given_theta(sample, model) for model in out_models]
    return np.mean(probs)

def pr_x(sample: MembershipDataset, out_models: list[torch.nn.Module], a: float) -> float:
    return 0.5 *((1+a)*pr_x_out(sample, out_models)+(1-a))

def pr_x_online(sample: MembershipDataset, models: list[torch.nn.Module]) -> float:
    return 0.5 * pr_x_out(sample, models)

def ratio_x(sample: MembershipDataset, out_models: list[torch.nn.Module], target_model: torch.nn.Module, a: float) -> float:
    return pr_x_given_theta(sample, target_model) / pr_x(sample, out_models, a)

def ratio_x_online(sample: MembershipDataset, models: list[torch.nn.Module], target_model: torch.nn.Module) -> float:
    return pr_x_given_theta(sample, target_model) / pr_x_online(sample, models)

def pr_z_batch(distribution_data: MembershipDataset, out_models: list[torch.nn.Module]) -> torch.Tensor:
    """Calculate the probabilities of the distribution dataset given the models in parallel."""
    probs = []
    for model in out_models:
        model.eval()
        model.to(device)
        imgs = torch.stack([img for _, img, _, _ in distribution_data]).to(device)
        labels = torch.tensor([label for _, _, label, _ in distribution_data]).to(device)
        with torch.no_grad():
            logits = model(imgs)
            model_probs = torch.nn.functional.softmax(logits, dim=1)
            probs.append(model_probs[range(len(labels)), labels])
    return torch.mean(torch.stack(probs), dim=0)

def ratio_z_batch(distribution_data: MembershipDataset, out_models: list[torch.nn.Module], target_model: torch.nn.Module) -> torch.Tensor:
    """Calculate the ratio of the probabilities of the distribution dataset given the target model and out models in parallel."""
    target_model.eval()
    target_model.to(device)
    imgs = torch.stack([img for _, img, _, _ in distribution_data]).to(device)
    labels = torch.tensor([label for _, _, label, _ in distribution_data]).to(device)
    with torch.no_grad():
        logits = target_model(imgs)
        target_probs = torch.nn.functional.softmax(logits, dim=1)
        target_probs = target_probs[range(len(labels)), labels]
    pr_z_vals = pr_z_batch(distribution_data, out_models)
    return target_probs / pr_z_vals

def rmia_online_score(sample: MembershipDataset, models: list[torch.nn.Module], target_model: torch.nn.Module, z: MembershipDataset, gamma: float=2) -> float:
    ratio_x_val = ratio_x_online(sample, models, target_model)
    ratio_z_vals = ratio_z_batch(z, models, target_model)
    count = torch.sum(ratio_x_val / ratio_z_vals > gamma).item()
    return count / len(distribution_data)

def get_rmia_online_score(test_dataset: MembershipDataset, target_model: torch.nn.Module, distribution_data: MembershipDataset, z: MembershipDataset, gamma: float=2, k: int=2) -> float:
    """Return scores for the test dataset. The score is calculated using the RMIA formula."""
    scores = []
    for i in tqdm(range(len(test_dataset)), desc="Calculating RMIA score"):
        sample = get_sample(test_dataset, i)
        models = create_online_models(sample, distribution_data)
        score = rmia_online_score(sample, models, target_model, z, gamma)
        scores.append(score)
    return scores

def select_closest_embeddings(distribution_data: MembershipDataset, test_dataset: MembershipDataset, n: int) -> list[int]:
    """From the distribution data, select the n closest embeddings to the test dataset embedding distribution.
    First for each sample in distribution data, calculate the average distance to all samples in the test dataset.
    Then select the n samples with the lowest average distance.
    """
    test_embeddings = np.array([img for _, img, _, _ in test_dataset])
    distribution_embeddings = np.array([img for _, img, _, _ in distribution_data])
    avg_distances = []
    for i in range(len(distribution_data)):
        avg_distance = np.mean(np.linalg.norm(test_embeddings - distribution_embeddings[i], axis=1))
        avg_distances.append(avg_distance)
    indexes = np.argsort(avg_distances)[:n]
    return indexes

def rmia_online_attack(distribution_data: MembershipDataset, test_dataset: MembershipDataset,k: int, num_z: int, gamma: float=2) -> list[float]:
    """Perform the RMIA attack. The attack is performed by splitting the distribution dataset into k parts, training
    k models on each part, and calculating the RMIA score for the test dataset. The RMIA score is calculated using the
    RMIA formula.
    """
    # only select samples with membership 1 as the distribution data
    distribution_data = get_sample(distribution_data, np.where(np.array(distribution_data.membership) == 1)[0])

    # load the target model
    target_model = load_target_model()
    #only pick num_z random samples from the distribution data. if num_z is -1, use all samples
    if num_z != -1:
        indexes = np.random.choice(len(distribution_data), num_z, replace=False)
        z = get_sample(distribution_data, indexes)

    # calculate the RMIA score
    scores = get_rmia_online_score(test_dataset, target_model, distribution_data, z, gamma, k)

    # save the scores
    temp_scores = torch.tensor(scores).to("cpu")
    torch.save(temp_scores, "rmia_online_scores.pt")

    return scores

def metrics(predicted_scores: np.array, ground_truth: np.array) -> float:
    fpr, tpr, _ = roc_curve(ground_truth, predicted_scores)
    tpr_at_fpr = np.interp(0.05, fpr, tpr)
    return tpr_at_fpr
        
if __name__ == "__main__":
    k = 32
    num_z = 2500
    a = 0.5
    gamma = 2
    num_runs = 100

    distribution_data: MembershipDataset = torch.load("data/pub.pt")
    test_dataset: MembershipDataset = torch.load("data/priv_out.pt")
    train_dataset: MembershipDataset = torch.load("data/train.pt")
    val_dataset: MembershipDataset = torch.load("data/val.pt")

    # cross validation
    # tpr_at_fpr, std = cross_val(distribution_data, k, a, num_z, gamma, num_runs)
    # print(f"TPR at FPR = 0.05: {tpr_at_fpr} +/- {std}")

    # validation
    # scores = rmia_online_attack(train_dataset, val_dataset, k, num_z, gamma)
    # ground_truth = np.array(val_dataset.membership)
    # predicted_scores = np.array(scores)
    # tpr_at_fpr = metrics(scores, ground_truth)
    # print(f"TPR at FPR = 0.05: {tpr_at_fpr}")


    scores = rmia_online_attack(distribution_data, test_dataset, k, num_z, gamma)
    df = pd.DataFrame({"ids": test_dataset.ids, "score": scores})
    df.to_csv(f"rmia_offline_scores_a_{a}.csv", index=False)


import os
import numpy as np
import pandas as pd
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


def sample_datasets(source_dataset: MembershipDataset, k: int) -> tuple[MembershipDataset]:
    """Split the source dataset into k equal parts. If k is 1, return the entire dataset as a single element tuple.
    Otherwise, create k empty MembershipDatasets and iterate through the source dataset, appending each item to each
    of the k datasets with a probability of 0.5. This way we ensure that the items are split evenly between the k datasets.
    """
    if k == 1:
        return (source_dataset,)
    
    datasets = [MembershipDataset() for _ in range(k)]
    for i in range(len(source_dataset)):
        sample = get_sample(source_dataset, i)
        for j in range(k):
            if np.random.rand() < 0.5:
                datasets[j] = append_sample(datasets[j], sample)
    return tuple(datasets)

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

def create_models_per_dataset(dataset: MembershipDataset, n: int=8, path: str="models"):
    """Create n models and train them on the dataset. The models are trained using the train_resnet18 function
    with variable epochs. The models are stored in a list and returned.
    """
    dataset_hash = create_dataset_hash(dataset)
    dataset_hash = dataset_hash[1:9]
    epoch_list = [1, 2, 5, 10, 15, 20]*n
    for epoch in epoch_list:
        model = train_resnet18(dataset, epoch, apply_transforms=False)
        model_hash = create_model_hash(model)
        model_hash = model_hash[1:9]
        torch.save(model, f"{path}/reference_{dataset_hash}_{model_hash}.pt")

def create_models(datasets: tuple[MembershipDataset], n: int=8, path: str="models"):
    """Create n models for each dataset in the datasets tuple. The models are trained using the train_resnet18 function
    with variable epochs. The models are stored in a list and returned.
    """
    for dataset in datasets:
        create_models_per_dataset(dataset, n)

def create_reference_models(source_dataset: MembershipDataset, k: int=1, n: int=8, path: str="models"):
    """Create k datasets from the source dataset and create n models for each dataset. The models are trained using the
    train_resnet18 function with variable epochs. The models are stored in a list and returned.
    """
    datasets = sample_datasets(source_dataset, k)
    create_models(datasets, n, path)

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
    return 0.5*((1 + a) * pr_x_out(sample, out_models) + (1 - a))

def ratio_x(sample: MembershipDataset, out_models: list[torch.nn.Module], target_model: torch.nn.Module, a: float) -> float:
    return pr_x_given_theta(sample, target_model) / pr_x(sample, out_models, a)

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

def rmia_score(sample: MembershipDataset, out_models: list[torch.nn.Module], target_model: torch.nn.Module, a: float, distribution_data: MembershipDataset, gamma: float=2) -> float:
    count = 0
    ratio_x_val = ratio_x(sample, out_models, target_model, a)
    ratio_z_vals = ratio_z_batch(distribution_data, out_models, target_model)
    count = torch.sum(ratio_x_val / ratio_z_vals > gamma).item()
    return count / len(distribution_data)

def get_rmia_score(test_dataset: MembershipDataset, out_models: list[torch.nn.Module], target_model: torch.nn.Module, a: float, distribution_data: MembershipDataset, gamma: float=2) -> float:
    """Return scores for the test dataset. The score is calculated using the RMIA formula."""
    scores = []
    for i in tqdm(range(len(test_dataset)), desc="Calculating RMIA score"):
        sample = get_sample(test_dataset, i)
        score = rmia_score(sample, out_models, target_model, a, distribution_data, gamma)
        scores.append(score)
    return scores

if __name__ == "__main__":
    # source_dataset: MembershipDataset = torch.load("data/pub.pt")
    # print(source_dataset.ids[0])
    # create_reference_models(source_dataset, k=1, n=4)

    # get list of model names that starts wih reference
    model_names = [name for name in os.listdir("models") if name.startswith("reference")]
    #randomly shuffle the model names and select 8 models
    np.random.shuffle(model_names)
    model_names = model_names[:8]
    # load the models
    out_models = [torch.load(f"models/{name}") for name in model_names]
    # load the target model
    target_model = load_target_model()
    # load the test dataset
    test_dataset: MembershipDataset = torch.load("data/priv_out.pt")
    # load the distribution dataset
    distribution_data: MembershipDataset = torch.load("data/pub.pt")
    #only pick 250 random samples from the distribution data
    indexes = np.random.choice(len(distribution_data), 500)
    distribution_data = get_sample(distribution_data, indexes)
    # calculate the RMIA score
    for a in [1]:
        scores = get_rmia_score(test_dataset, out_models, target_model, a, distribution_data)
        df = pd.DataFrame({"ids": test_dataset.ids, "score": scores})
        df.to_csv(f"rmia_offline_scores_a_{a}.csv", index=False)
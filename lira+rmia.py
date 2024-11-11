from model import load_target_model, train_resnet18
from preprocess import MembershipDataset
from infer import infer
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm


def sample_datasets(source_dataset: MembershipDataset) -> tuple[MembershipDataset]:
    n = len(source_dataset) // 8
    datasets = []
    for i in range(8):
        dataset = MembershipDataset()
        dataset.ids = source_dataset.ids[i * n : (i + 1) * n]
        dataset.imgs = source_dataset.imgs[i * n : (i + 1) * n]
        dataset.labels = source_dataset.labels[i * n : (i + 1) * n]
        dataset.membership = source_dataset.membership[i * n : (i + 1) * n]
        datasets.append(dataset)
    return tuple(datasets)


def split_dataset(
    dataset: MembershipDataset, ratio: int = 0.8
) -> tuple[MembershipDataset, MembershipDataset]:
    n = int(len(dataset) * ratio)
    train = MembershipDataset()
    test = MembershipDataset()
    train.ids = dataset.ids[:n]
    train.imgs = dataset.imgs[:n]
    train.labels = dataset.labels[:n]
    train.membership = dataset.membership[:n]
    test.ids = dataset.ids[n:]
    test.imgs = dataset.imgs[n:]
    test.labels = dataset.labels[n:]
    test.membership = dataset.membership[n:]
    return train, test


def sample_z(dataset: MembershipDataset, x_index: int) -> int:
    z_index = x_index
    while z_index == x_index:
        z_index = np.random.randint(0, len(dataset) - 1)
    return z_index


def pop_sample(dataset: MembershipDataset, index: int) -> MembershipDataset:
    sample = MembershipDataset()
    sample.ids = [dataset.ids[index]]
    sample.imgs = [dataset.imgs[index]]
    sample.labels = [dataset.labels[index]]
    sample.membership = [dataset.membership[index]]

    new_dataset = MembershipDataset()
    new_dataset.ids = dataset.ids[:index] + dataset.ids[index + 1 :]
    new_dataset.imgs = dataset.imgs[:index] + dataset.imgs[index + 1 :]
    new_dataset.labels = dataset.labels[:index] + dataset.labels[index + 1 :]
    new_dataset.membership = (
        dataset.membership[:index] + dataset.membership[index + 1 :]
    )

    return sample, new_dataset


def append_sample(
    dataset: MembershipDataset, sample: MembershipDataset
) -> MembershipDataset:
    new_dataset = MembershipDataset()
    new_dataset.ids = dataset.ids + sample.ids
    new_dataset.imgs = dataset.imgs + sample.imgs
    new_dataset.labels = dataset.labels + sample.labels
    new_dataset.membership = dataset.membership + sample.membership
    return new_dataset


def create_x_z_datasets(
    source_dataset: MembershipDataset, x_index: int, z_index: int
) -> tuple[list[MembershipDataset], list[MembershipDataset]]:
    x_sample, source_dataset = pop_sample(source_dataset, x_index)
    z_sample, source_dataset = pop_sample(source_dataset, z_index)
    sampled_datasets = sample_datasets(source_dataset)
    x_datasets = [append_sample(dataset, x_sample) for dataset in sampled_datasets]
    z_datasets = [append_sample(dataset, z_sample) for dataset in sampled_datasets]
    return x_datasets, z_datasets


def train_models(dataset: MembershipDataset) -> list[torch.nn.Module]:
    models = []
    for i in [10, 15, 20]:
        model = train_resnet18(dataset, epochs=i)
        models.append(model)
    return models


def create_x_z_models(
    x_datasets: list[MembershipDataset], z_datasets: list[MembershipDataset]
) -> tuple[list[torch.nn.Module], list[torch.nn.Module]]:
    x_models = []
    z_models = []
    for x_dataset, z_dataset in zip(x_datasets, z_datasets):
        x_models.extend(train_models(x_dataset))
        z_models.extend(train_models(z_dataset))
    return x_models, z_models


def get_x_z_scores(
    x_models: list[torch.nn.Module],
    z_models: list[torch.nn.Module],
    sample: MembershipDataset,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    x_scores = [
        infer(model, sample)[f"score_class_{sample.labels[0]}"].values
        for model in x_models
    ]
    z_scores = [
        infer(model, sample)[f"score_class_{sample.labels[0]}"].values
        for model in z_models
    ]
    x_scores = [torch.tensor(score) for score in x_scores]
    z_scores = [torch.tensor(score) for score in z_scores]
    return x_scores, z_scores


def get_x_z_mean_var(
    x_scores: list[torch.Tensor], z_scores: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    x_mean = torch.stack(x_scores).mean(dim=0)
    x_var = torch.stack(x_scores).var(dim=0)
    z_mean = torch.stack(z_scores).mean(dim=0)
    z_var = torch.stack(z_scores).var(dim=0)
    return x_mean, x_var, z_mean, z_var


def get_likelihood(
    score: torch.Tensor, mean: torch.Tensor, var: torch.Tensor
) -> torch.Tensor:
    return torch.exp(-0.5 * ((score - mean) ** 2) / var) / torch.sqrt(2 * np.pi * var)


def get_likelihood_ratio(
    score: torch.Tensor,
    x_mean: torch.Tensor,
    x_var: torch.Tensor,
    z_mean: torch.Tensor,
    z_var: torch.Tensor,
) -> torch.Tensor:
    x_likelihood = get_likelihood(score, x_mean, x_var)
    z_likelihood = get_likelihood(score, z_mean, z_var)
    return x_likelihood / z_likelihood


def pairwise_likelihood_ratio(
    data: MembershipDataset, x_index: int, z_index: int, target_model: torch.nn.Module
) -> torch.Tensor:
    x_datasets, z_datasets = create_x_z_datasets(data, x_index, z_index)
    x_models, z_models = create_x_z_models(x_datasets, z_datasets)
    sample, _ = pop_sample(data, x_index)
    score = infer(target_model, sample)[f"score_class_{sample.labels[0]}"].values
    score = torch.tensor(score)
    x_scores, z_scores = get_x_z_scores(x_models, z_models, sample)
    x_mean, x_var, z_mean, z_var = get_x_z_mean_var(x_scores, z_scores)
    return get_likelihood_ratio(score, x_mean, x_var, z_mean, z_var)


def likelihood_ratio_attack_per_sample(
    data: MembershipDataset, x_index: int, target_model: torch.nn.Module, gamma: float
) -> torch.Tensor:
    likelihood_ratios = []
    z_indices = [sample_z(data, x_index) for _ in range(50)]
    for z_index in z_indices:
        likelihood_ratio = pairwise_likelihood_ratio(
            data, x_index, z_index, target_model
        )
        likelihood_ratios.append(likelihood_ratio)
    likelihood_ratios = torch.stack(likelihood_ratios)
    return (likelihood_ratios > gamma).float().mean()


def likelihood_ratio_attack(
    data: MembershipDataset, target_model: torch.nn.Module
) -> torch.Tensor:
    scores = []
    for i in tqdm(range(len(data)), desc="attacking samples"):
        score = likelihood_ratio_attack_per_sample(data, i, target_model, 1)
        scores.append(score)
    return torch.stack(scores)


def save_results_to_csv(data: MembershipDataset, scores: torch.Tensor, path: str):
    df = pd.DataFrame(
        {
            "ids": data.ids,
            "membership": data.membership,
            "score": scores,
        }
    )
    df.to_csv(path, index=False)


if __name__ == "__main__":
    target_model = load_target_model()
    data: MembershipDataset = torch.load("data/pub.pt")
    results = likelihood_ratio_attack(data, target_model)
    save_results_to_csv(data, results, "train_results.csv")

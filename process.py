from preprocess import PseudoLabelDataset

def append_sample(dataset: PseudoLabelDataset, sample: PseudoLabelDataset) -> PseudoLabelDataset:
    """Append the sample dataset to the dataset. This is done by concatenating the ids, imgs, labels, and membership
    of the sample dataset to the dataset.
    """
    new_dataset = PseudoLabelDataset()
    new_dataset.ids = dataset.ids + sample.ids
    new_dataset.imgs = dataset.imgs + sample.imgs
    new_dataset.labels = dataset.labels + sample.labels
    new_dataset.membership = dataset.membership + sample.membership
    new_dataset.pseudo_labels = dataset.pseudo_labels + sample.pseudo_labels
    return new_dataset

def pop_sample(dataset: PseudoLabelDataset, index: int) -> tuple[PseudoLabelDataset, PseudoLabelDataset]:
    """Pop the sample dataset at the index. Return the popped sample and the new dataset.
    """
    sample = get_sample(dataset, index)
    new_dataset = PseudoLabelDataset()
    new_dataset.ids = dataset.ids[:index] + dataset.ids[index+1:]
    new_dataset.imgs = dataset.imgs[:index] + dataset.imgs[index+1:]
    new_dataset.labels = dataset.labels[:index] + dataset.labels[index+1:]
    new_dataset.membership = dataset.membership[:index] + dataset.membership[index+1:]
    new_dataset.pseudo_labels = dataset.pseudo_labels[:index] + dataset.pseudo_labels[index+1:]
    return sample, new_dataset

def get_sample(dataset: PseudoLabelDataset, index: int | list[int]) -> PseudoLabelDataset:
    """Get the sample dataset at the index. This is done by creating a new PseudoLabelDataset and copying the ids, imgs,
    labels, and membership of the dataset at the index to the new dataset.
    """
    sample = PseudoLabelDataset()
    if isinstance(index, int):
        index = [index]
    sample.ids = [dataset.ids[i] for i in index]
    sample.imgs = [dataset.imgs[i] for i in index]
    sample.labels = [dataset.labels[i] for i in index]
    sample.membership = [dataset.membership[i] for i in index]
    sample.pseudo_labels = [dataset.pseudo_labels[i] for i in index]
    return sample

def get_population_dataset(dataset: PseudoLabelDataset) -> PseudoLabelDataset:
    """Get the population dataset. This is done by creating a new PseudoLabelDataset and copying the ids, imgs, labels
    of the samples with membership 1 to the new dataset.
    """
    population_dataset = PseudoLabelDataset()
    population_dataset.ids = [dataset.ids[i] for i in range(len(dataset)) if dataset.membership[i] == 1]
    population_dataset.imgs = [dataset.imgs[i] for i in range(len(dataset)) if dataset.membership[i] == 1]
    population_dataset.labels = [dataset.labels[i] for i in range(len(dataset)) if dataset.membership[i] == 1]
    population_dataset.membership = [dataset.membership[i] for i in range(len(dataset)) if dataset.membership[i] == 1]
    population_dataset.pseudo_labels = [dataset.pseudo_labels[i] for i in range(len(dataset)) if dataset.membership[i] == 1]
    return population_dataset
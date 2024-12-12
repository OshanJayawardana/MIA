import torch
from torch.utils.data import Dataset


class TaskDataset(Dataset):
    """Dataset for the task of predicting the class of an image."""

    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []

        self.transform = transform

    def __getitem__(self, index) -> tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    """Dataset for the task of predicting the class of an image"""

    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index) -> tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


# create a new dataset class which inherits from MembershipDataset but has both membership and pseudo-labels
class PseudoLabelDataset(TaskDataset):
    """Dataset for the task of predicting the class of an image and its pseudo-label"""

    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []
        self.pseudo_labels = []

    def __getitem__(self, index) -> tuple[int, torch.Tensor, int, int, int]:
        id_, img, label = super().__getitem__(index)
        pseudo_label = self.pseudo_labels[index]
        membership = self.membership[index]
        return id_, img, label, membership, pseudo_label

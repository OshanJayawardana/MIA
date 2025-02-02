from preprocess import MembershipDataset
from model import load_target_model
from torch.nn import Module
import pandas as pd
import torch
import os
from torchvision.models import resnet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def infer(model: Module, data: MembershipDataset) -> torch.Tensor:
    """Infer the model on the data.

    Args:
        model (Module): The model to infer.
        data (MembershipDataset): The data to infer on.
        path (str): The path to save the results.
    
    Returns:

    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        scores = []
        for _, img, _, _, _ in data:
            logits = model(img[None, ...].to(device))
            scores.append(logits.cpu())
        scores = torch.cat(scores)
    return scores

if __name__ == "__main__":
    model = load_target_model()
    data: MembershipDataset = torch.load("data/pub.pt")
    scores= infer(model, data)
    scores.to_csv("train_scores.csv", index=False)

    data: MembershipDataset = torch.load("data/priv_out.pt")
    scores = infer(model, data)
    scores.to_csv("test_scores.csv", index=False)
    
from preprocess import MembershipDataset, LogitstDataset
import torch
from model import load_target_model

model = load_target_model()

def append_logits(dataset: MembershipDataset) -> LogitstDataset:
    imgs = torch.stack(dataset.imgs)
    with torch.no_grad():
        outputs = model(imgs)
        logits = outputs.cpu().numpy()

    new_dataset = LogitstDataset()
    new_dataset.ids = dataset.ids
    new_dataset.imgs = dataset.imgs
    new_dataset.labels = dataset.labels
    new_dataset.membership = dataset.membership
    new_dataset.logits = logits
    return new_dataset

dataset_list = [("data/pub.pt", "data/pub_w_logits.pt"),
                ("data/priv_out.pt", "data/priv_out_w_logits.pt")]

for dataset_path, save_path in dataset_list:
    dataset = torch.load(dataset_path)
    dataset = append_logits(dataset)
    torch.save(dataset, save_path)
    
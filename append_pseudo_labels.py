from preprocess import MembershipDataset, PseudoLabelDataset
from model import load_target_model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_model = load_target_model()
target_model.eval()
target_model.to(device)

# given a Membership dataset infer the pseudo-labels using the target model. Append the pseudo-labels to the dataset
# and return the new PseudoLabelDataset
def append_pseudo_labels(dataset: MembershipDataset):
    pseudo_labels = []
    imgs = torch.stack(dataset.imgs).to(device)
    with torch.no_grad():
        outputs = target_model(imgs)
        preds = torch.argmax(outputs, dim=1)
        pseudo_labels = preds.cpu().numpy().tolist()
    
    new_dataset = PseudoLabelDataset()
    new_dataset.transform = dataset.transform
    new_dataset.ids = dataset.ids
    new_dataset.imgs = dataset.imgs
    new_dataset.labels = dataset.labels
    new_dataset.membership = dataset.membership
    new_dataset.pseudo_labels = pseudo_labels
    return new_dataset

dataset_list = [
    # ("data/train.pt", "data/train_w_pseudo.pt"), 
    ("data/pub.pt", "data/pub_w_pseudo.pt"),
    ("data/priv_out.pt", "data/priv_out_w_pseudo.pt"), 
    # ("data/val.pt", "data/val_w_pseudo.pt")
    ]

for dataset_path, save_path in dataset_list:
    dataset = torch.load(dataset_path)
    dataset = append_pseudo_labels(dataset)
    torch.save(dataset, save_path)
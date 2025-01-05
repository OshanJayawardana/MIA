from preprocess import PseudoLabelDataset
import torch 

pub: PseudoLabelDataset = torch.load("data/pub_w_pseudo.pt")
priv: PseudoLabelDataset = torch.load("data/priv_out_w_pseudo.pt")

# overwrite the labels of the private dataset with 35
priv.labels = [35 for _ in range(len(priv.labels))]
torch.save(priv, "data/priv_overwrite.pt")

# overwrite the labels of the public dataset with 35
pub.labels = [35 for _ in range(len(pub.labels))]
torch.save(pub, "data/pub_overwrite.pt")

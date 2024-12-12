import os
import numpy as np
import torch
from preprocess import MembershipDataset
from model import load_target_model
from infer import infer
from rmia_offline import metrics
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data: MembershipDataset = torch.load('data/train.pt')
val_data: MembershipDataset = torch.load('data/val.pt')

target_model = load_target_model()

# check if the file data/classifier_train.pt exist
if os.path.exists('data/classifier_train.pt'):
    train_data = torch.load('data/classifier_train.pt')
    train_inputs, train_img_labels, train_labels = train_data.tensors
else:
    # append logits to train and val data
    train_labels = torch.tensor(train_data.membership).to(device)
    train_logits = infer(target_model, train_data).to(device) # shape: (n_samples, 44)
    train_img_labels = torch.tensor(train_data.labels).to(device) # shape: (n_samples,)
    # save the classifier train dataset
    torch.save(TensorDataset(train_logits, train_img_labels, train_labels), 'data/classifier_train.pt')

# check if the file data/classifier_val.pt exist
if os.path.exists('data/classifier_val.pt'):
    val_data = torch.load('data/classifier_val.pt')
    val_inputs, val_img_labels, val_labels = val_data.tensors
else:
    val_labels = torch.tensor(val_data.membership).to(device)
    val_logits = infer(target_model, val_data).to(device)
    val_img_labels = torch.tensor(val_data.labels).to(device)
    # save the classifier val dataset
    torch.save(TensorDataset(val_logits, val_img_labels, val_labels), 'data/classifier_val.pt')

val_labels = val_labels.cpu().numpy()

# create a self attention based classifier. where the one hot of img_labels are keys, and the logits are values
class Classifier(nn.Module):
    def __init__(self, n_classes: int, n_features: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(n_features, 1)
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, img_labels, logits):
        keys = torch.eye(img_labels.max() + 1).to(device)[img_labels]
        keys = keys[None, ...].repeat(logits.shape[0], 1, 1)
        values = logits[None, ...]
        attn_output, _ = self.attention(keys, keys, values)
        return self.fc(attn_output[0])
    
classifier = Classifier(n_classes=2, n_features=44).to(device)
classifier.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(classifier.parameters(), lr=1e-3)

for epoch in range(10):
    classifier.train()
    preds = classifier(train_img_labels, train_logits)
    loss = criterion(preds, train_labels)
    loss.backward()
    optimizer.step()

    classifier.eval()
    with torch.no_grad():
        val_preds = classifier(val_inputs).argmax(dim=1)
    val_preds = val_preds.cpu().numpy()
    tpr_at_fpr = metrics(val_preds, val_labels)
    print(f"Epoch {epoch}: TPR@0.05FPR: {tpr_at_fpr}")




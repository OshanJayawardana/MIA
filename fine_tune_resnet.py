from model import load_target_model
from preprocess import MembershipDataset
import torch
import random
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_sample(dataset: MembershipDataset, indexes: list) -> MembershipDataset:
    new_dataset = MembershipDataset()
    new_dataset.ids = [dataset.ids[i] for i in indexes]
    new_dataset.imgs = [dataset.imgs[i] for i in indexes]
    new_dataset.labels = [dataset.labels[i] for i in indexes]
    new_dataset.membership = [dataset.membership[i] for i in indexes]
    return new_dataset

dataset = torch.load("data/pub.pt")
train_indexes = random.sample(range(len(dataset.labels)), int(0.5 * len(dataset.labels)))
val_indexes = [i for i in range(len(dataset.labels)) if i not in train_indexes]

train_dataset = get_sample(dataset, train_indexes)
val_dataset = get_sample(dataset, val_indexes)

X_train = torch.stack(train_dataset.imgs).to(device)
y_train = torch.tensor(train_dataset.membership).to(device)

X_val = torch.stack(val_dataset.imgs).to(device)
y_val = torch.tensor(val_dataset.membership).to(device)

model = load_target_model()
# attach new fully connected layer for binary classification
model.fc = torch.nn.Linear(512, 1)
model.to(device)

# only the new fully connected layer is trained
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.1)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs.squeeze(), y_train.float())
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs.squeeze(), y_val.float())
        val_preds = torch.round(torch.sigmoid(val_outputs))

    # threshold by 0.5 and calculate accuracy
    correct = (val_preds.squeeze() == y_val).sum().item()
    print(val_preds)
    accuracy = correct / len(X_val)

    print(f"Epoch {epoch + 1}/10, Loss: {loss.item()}, Val Loss: {val_loss.item()}")
    print(f"Val Accuracy: {accuracy}")

# X_train = torch.stack(dataset.imgs).to(device)
# y_train = torch.tensor(dataset.membership).to(device)

# model = load_target_model()
# model.fc = torch.nn.Linear(512, 1)
# model.to(device)

# criterion = torch.nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.01)

# for epoch in range(12):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train)
#     loss = criterion(outputs.squeeze(), y_train.float())
#     loss.backward()
#     optimizer.step()

#     print(f"Epoch {epoch + 1}/10, Loss: {loss.item()}")

# test_dataset : MembershipDataset = torch.load("data/priv_out.pt")
# ids = test_dataset.ids
# imgs = torch.stack(test_dataset.imgs).to(device)
# model.eval()
# with torch.no_grad():
#     outputs = model(imgs)

# score = outputs.squeeze().cpu().numpy()
# # save ids and score to a pandas dataframe
# df = pd.DataFrame({"ids": ids, "score": score})
# df.to_csv("results/finetune.csv", index=False)
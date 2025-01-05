from preprocess import MembershipDataset, GradientDataset
from model import load_target_model
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_target_model()
model.eval()
model.to(device)


def append_gradients(dataset: MembershipDataset) -> GradientDataset:
    """
    For each sample append the gradients of the last fully connected layer of the target model to the dataset.
    """
    imgs = torch.stack(dataset.imgs).to(device)
    labels = torch.tensor(dataset.labels).to(device)

    # List to store gradients
    gradients = []

    for img, label in tqdm(zip(imgs, labels)):
        img = img.unsqueeze(0)
        label = label.unsqueeze(0)

        # Forward pass
        model.zero_grad()
        outputs = model(img)
        loss = torch.nn.CrossEntropyLoss()(outputs, label)
        loss.backward()

        # Extract gradients of the last fully connected layer
        last_fc_gradients = None
        for name, param in model.named_parameters():
            if "fc" in name and param.requires_grad:
                last_fc_gradients = param.grad.view(-1).to("cpu")

        if last_fc_gradients is not None:
            gradients.append(last_fc_gradients)

    # Append gradients to the dataset
    new_dataset = GradientDataset()
    new_dataset.ids = dataset.ids
    new_dataset.imgs = dataset.imgs
    new_dataset.labels = dataset.labels
    new_dataset.membership = dataset.membership
    new_dataset.gradients = gradients

    return new_dataset


dataset_list = [("data/pub.pt", "data/pub_w_gradients.pt"), ("data/priv_out.pt", "data/priv_out_w_gradients.pt")]

for dataset_path, save_path in dataset_list:
    dataset = torch.load(dataset_path)
    new_dataset = append_gradients(dataset)
    torch.save(new_dataset, save_path)

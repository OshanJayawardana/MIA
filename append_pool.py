from model import load_target_model
from preprocess import MembershipDataset, LogitstDataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# append the output of the avgpool layer to the dataset
def append_pool(dataset: MembershipDataset, model: torch.nn.Module) -> LogitstDataset:
    """
    For each sample in the dataset append the output of the avgpool layer to the dataset.
    """
    model.eval()
    model.to(device)
    imgs = torch.stack(dataset.imgs).to(device)
    
    # create a forward hook to store the output of the avgpool layer
    def hook(module, input, output):
        hook.output = output

    model.avgpool.register_forward_hook(hook)
    with torch.no_grad():
        model(imgs)

    # append the output of the avgpool layer to the dataset
    new_dataset = LogitstDataset()
    new_dataset.ids = dataset.ids
    new_dataset.imgs = dataset.imgs
    new_dataset.labels = dataset.labels
    new_dataset.membership = dataset.membership
    new_dataset.logits = hook.output.cpu().squeeze().numpy()

    return new_dataset  

# load the target model
model = load_target_model()

# load the dataset
dataset = torch.load("data/priv_out.pt")

# append the output of the avgpool layer to the dataset
new_dataset = append_pool(dataset, model)

# save the new dataset
torch.save(new_dataset, "data/priv_out_w_pool.pt")
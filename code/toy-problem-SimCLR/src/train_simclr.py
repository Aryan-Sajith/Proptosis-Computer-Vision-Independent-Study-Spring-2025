import yaml, torch, argparse, os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from models import SimCLRModel
from datasets import UTKFaceDataset
from utils import nt_xent_loss

def get_simclr_transforms():
    """Strong augmentations for SimCLR pre-training"""
    color_jitter = transforms.ColorJitter(
        brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
    ])

# Custom dataset to generate two views of each image
class TwoViewsDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
        
    def __getitem__(self, idx):
        img = self.base_dataset[idx]
        view1 = self.transform(img)
        view2 = self.transform(img)  # Different random augmentations due to randomness in transform
        return view1, view2
    
    def __len__(self):
        return len(self.base_dataset)

def train(cfg):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load base dataset (without transforms)
    base_ds = UTKFaceDataset(cfg['data']['root'], cfg['data']['splits'], 'ssl', transform=None)
    
    # Wrap with two views dataset that applies transforms twice
    ds = TwoViewsDataset(base_ds, get_simclr_transforms())
    
    # Create data loader
    loader = DataLoader(ds, cfg['train']['batch_size'], shuffle=True, num_workers=4, drop_last=True)
    
    # Initialize model
    model = SimCLRModel(base_arch=cfg['model']['arch'],
                        proj_hidden=cfg['model']['proj_hidden'],
                        proj_out=cfg['model']['proj_out']).to(device)
    
    # Setup optimizer
    opt = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])

    # Create checkpoint directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)

    # Training loop
    print("Training SimCLR...")
    for epoch in range(cfg['train']['epochs']):
        # Track total loss for this epoch
        total_loss = 0

        # Iterate over batches
        for view1, view2 in loader:
            # Move data to device
            view1, view2 = view1.to(device), view2.to(device)
            
            # Forward pass through the model to get projections
            _, z1 = model(view1)
            _, z2 = model(view2)
            
            # Compute NT-Xent loss between the two views
            loss = nt_xent_loss(z1, z2, cfg['train']['temp'])
            
            # Backpropagation
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
        # Print average loss for this epoch    
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

        # Save model checkpoint every save_freq epochs
        if (epoch+1) % cfg['train']['save_freq'] == 0:
            torch.save(model.state_dict(), f"checkpoints/simclr_ep{epoch+1}.pt")
    
    # Save final model
    torch.save(model.state_dict(), "checkpoints/simclr_final.pt")
    print("SimCLR training complete!")

if __name__=='__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    # Add config file argument
    parser.add_argument('--cfg', default='configs/simclr.yaml')
    # Add other arguments as needed
    args = parser.parse_args()
    # Load config file
    cfg = yaml.safe_load(open(args.cfg))
    # Train the model
    train(cfg)
import yaml, torch, argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from models import SimCLRModel, Classifier
from datasets import UTKFaceDataset
from utils import accuracy
from tqdm import tqdm

def get_train_transforms():
    # Data augmentation for training (includes random transforms)
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

def get_val_transforms():
    # Deterministic transforms for validation (no random augmentations)
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

def train(cfg, mode):
    """mode: 'scratch' or 'ssl_ft'"""
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    # Obtain training and validation datasets after indexing the proper splits
    ds_train = UTKFaceDataset(cfg['data']['root'], cfg['data']['splits'], 'train', transform=get_train_transforms())
    ds_val   = UTKFaceDataset(cfg['data']['root'], cfg['data']['splits'], 'val',   transform=get_val_transforms())
    dl_train = DataLoader(ds_train, cfg['train']['batch_size'], shuffle=True)
    dl_val   = DataLoader(ds_val,   cfg['train']['batch_size'], shuffle=False)

    # encoder
    simclr = SimCLRModel(base_arch=cfg['model']['arch']).to(device)

    # Load pretrained weights if mode is 'ssl_ft'
    if mode == 'ssl_ft':
        simclr.load_state_dict(torch.load(cfg['train']['pretrained_ckpt']))
    encoder = simclr.encoder

    # Freeze encoder parameters for fine-tuning
    # or unfreeze them for scratch training
    if mode == 'scratch':
        # Freeze encoder parameters for scratch training
        for p in encoder.parameters(): p.requires_grad = True
    else: # Unfreeze encoder parameters for fine-tuning
        for p in encoder.parameters(): p.requires_grad = False

    # Defines the classifier head to be trained on top of the encoder
    # The classifier head is a linear layer that maps the encoder output to the number of classes
    # The encoder output is the feature vector of the input image
    clf = Classifier(feat_dim=cfg['model']['feat_dim'], num_classes=2).to(device)
    opt = torch.optim.Adam(list(filter(lambda p: p.requires_grad, encoder.parameters())) + list(clf.parameters()),
                            lr=cfg['train']['lr'])
    
    # Track the best accuracy
    best_acc = 0.0

    # Training loop
    for epoch in range(cfg['train']['epochs']):
        # Train the model
        encoder.train(); clf.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}", unit="batch") 

        # Define total loss for this epoch
        for x,y in dl_train:
            # Move data to device which can be either CPU or GPU
            x,y = x.to(device), y.to(device)
            # Extract features from the input images using the encoder
            h = encoder(x).squeeze()
            # Compute Cross-Entropy loss
            out = clf(h)
            loss = torch.nn.functional.cross_entropy(out, y)
            # Backpropagation: Zero gradients, compute loss, and update weights
            opt.zero_grad(); loss.backward(); opt.step()
            # Update the progress bar with the loss
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)
        
        # Sets both the encoder and classifier to evaluation mode
        # This is important for dropout and batch normalization layers
        # to behave differently during training and evaluation
        # The encoder is used to extract features from the input images
        # The classifier is used to predict the class labels from the features
        encoder.eval(); clf.eval()
        
        # Track the accuracy on the validation set
        acc = 0

        # Iterate over the validation set
        with torch.no_grad():

            # Iterate over batches
            for x,y in dl_val:
                # Move data to device
                x,y = x.to(device), y.to(device)
                # Extract features using the encoder
                h = encoder(x).squeeze()
                # Compute the accuracy of the classifier
                acc += accuracy(clf(h), y).item()
        # Normalize the accuracy by the number of batches
        acc /= len(dl_val)
        print(f"[{mode}] Epoch {epoch} val_acc={acc:.4f}")
        
        # Save the model if the accuracy is better than the best accuracy
        if acc > best_acc:
            best_acc = acc
            torch.save({'enc': encoder.state_dict(), 'clf': clf.state_dict()},
                       f"checkpoints/{mode}_best.pt")

if __name__=='__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    # Add arguments for configuration file and mode
    parser.add_argument('--cfg', default='configs/supervised.yaml')
    # Add argument for mode (scratch or ssl_ft)
    parser.add_argument('--mode', choices=['scratch','ssl_ft'], required=True)
    # Parse the arguments
    args = parser.parse_args()
    # Load the configuration file
    cfg = yaml.safe_load(open(args.cfg))
    # Train the model
    train(cfg, args.mode)

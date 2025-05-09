import torch.nn as nn
import torchvision.models as tv

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x): return self.net(x)

class SimCLRModel(nn.Module):
    """Encoder + projection head for contrastive pre‐training."""
    def __init__(self, base_arch='resnet18', proj_hidden=512, proj_out=128):
        super().__init__()
        backbone = tv.__dict__[base_arch](pretrained=False)
        # remove final fc
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features
        self.projection = ProjectionHead(feat_dim, proj_hidden, proj_out)

    def forward(self, x):
        h = self.encoder(x).squeeze()   # shape [B, feat_dim]
        z = self.projection(h)          # shape [B, proj_out]
        return h, z

class Classifier(nn.Module):
    """Linear head for few-shot classification."""
    def __init__(self, feat_dim=512, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
    def forward(self, x): return self.fc(x)

import torch, torch.nn.functional as F

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Compute NT-Xent loss between two batches of projections.
    Args:
        z_i: projections from the first augmentation, shape [B, D]
        z_j: projections from the second augmentation, shape [B, D]
        temperature: temperature parameter for scaling similarity
    Returns:
        loss: computed NT-Xent loss (scalar)
    """
    # Concatenate both batches and normalize
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
    z = F.normalize(z, dim=1)  # normalize along feature dimension
    
    # Compute cosine similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # [2B, 2B]
    
    # Create mask for positive pairs
    # For each i in 0...B-1, the positive pair is (i, i+B) and (i+B, i)
    sim_mask = torch.zeros_like(sim)
    sim_mask[torch.arange(batch_size), torch.arange(batch_size, 2*batch_size)] = 1
    sim_mask[torch.arange(batch_size, 2*batch_size), torch.arange(batch_size)] = 1
    
    # For the denominator, we need all pairs except self-similarity (diagonal)
    mask = ~torch.eye(2*batch_size, dtype=torch.bool, device=z.device)
    
    # Apply masks to get positive and negative similarity scores
    pos_sim = torch.sum(sim * sim_mask, dim=1)  # [2B]
    neg_sim = torch.exp(sim) * mask  # [2B, 2B] with zeros on diagonal
    
    # Compute loss: -log( exp(pos_sim) / sum(exp(neg_sim)) )
    loss = -torch.log(torch.exp(pos_sim) / neg_sim.sum(dim=1))
    
    return loss.mean()  # average over batch


def accuracy(output, target):
    preds = output.argmax(dim=1) # get predicted classes
    return (preds == target).float().mean() # compute accuracy

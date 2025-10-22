import torch


class TripletLoss(torch.nn.Module):
    """Basic triplet loss

    Does not build triplet online
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.triplet_used_proportion = 0.0

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """Build a triplet loss from anchor positive and negative values.

        Args
            anchor (Tensor): Anchor for each triplet. Expected shape: B x n
            positive (Tensor): Positive samples. Expected shape: B x n
            negative (Tensor): Negative samples. Expected shape: B x n

        Returns
            Tensor: Averaged triplet loss
        """
        distance_positive = (anchor - positive).pow(2).sum(dim=1)
        distance_negative = (anchor - negative).pow(2).sum(dim=1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        # losses = distance_positive + torch.relu(-distance_negative + self.margin) # Similar to contrastive loss

        self.triplet_used_proportion = (losses.count_nonzero() / losses.numel()).item()
        # self.triplet_used_proportion = ((distance_negative < self.margin).sum() / distance_negative.numel()).item()

        return losses.mean()

import torch
import torch.nn as nn

from config import SEED

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LinearProbe(nn.Module):
    """
    A linear probe: a single linear layer followed by softmax.
    The softmax is applied by the loss function during training and by the predict method during inference.
    """

    def __init__(self, d_in, n_classes):
        super().__init__()
        self.linear = nn.Linear(d_in, n_classes)

    def forward(self, x):
        return self.linear(x)

    @torch.no_grad()
    def predict(self, x):
        return self(x).softmax(dim=-1).argmax(dim=-1)


def standardize(x_train, x_test):
    """
    Standardize features with the train statistics only (no leakage from the test set).

    Args:
        x_train: Tensor of shape [N_train, d]
        x_test: Tensor of shape [N_test, d]

    Returns:
        x_train, x_test standardized
    """

    mean, std = x_train.mean(dim=0), x_train.std(dim=0).clamp_min(1e-6)
    return (x_train - mean) / std, (x_test - mean) / std


def train_probe(x, y, n_classes, epochs, lr, l2, seed=SEED):
    """
    Train a linear probe on the given features and labels (full batch, Adam).

    Args:
        x: Tensor of shape [N, d], where N is the number of samples and d is the feature dimension.
        y: Tensor of shape [N], containing the integer labels for each sample.
        n_classes: number of classes
        epochs: number of training epochs
        lr: learning rate for Adam
        l2: weight of the L2 penalty on W added to the loss
        seed: random seed for the weight initialization

    Returns:
        probe: the trained LinearProbe model
    """

    torch.manual_seed(seed)
    probe = LinearProbe(x.shape[1], n_classes).to(x.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    probe.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(probe(x), y) + l2 * probe.linear.weight.pow(2).sum()
        loss.backward()
        optimizer.step()
    probe.eval()
    return probe

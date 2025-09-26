import torch
from torch import nn


def neg_stack_logits(logits: torch.Tensor):
    return torch.stack([logits, -logits], dim=-1)


class StackedModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return neg_stack_logits(self.model(*args, **kwargs))

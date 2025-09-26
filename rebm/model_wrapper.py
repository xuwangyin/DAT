import torch
from torch import nn


def neg_stack_logits(logits: torch.Tensor):
    return torch.stack([logits, -logits], dim=-1)


def stack_logit(logits: torch.Tensor):
    return torch.stack([logits], dim=-1)


class StackedModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model
        self.last_layer = nn.Identity()

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        out = self.last_layer(out)
        # print("out.shape", out.shape)
        # return sigmoid(out)
        return out
        # return stack_logit(out)

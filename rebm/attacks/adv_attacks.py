import torch
import torch.nn.functional as F
from torch import nn

from rebm.attacks.attack_steps import L2Step, LinfStep


def adam_attack(
    model: nn.Module,
    x: torch.Tensor,
    eps: float = 0.01,
    steps: int = 50,  # default is for generation
    lr: float | None = 5e-3,
    betas: tuple = (0.9, 0.999),
    random_start: bool = False,
    descent: bool = False,
) -> torch.Tensor:
    assert not model.training
    assert not x.requires_grad

    if steps == 0:
        return x

    delta = torch.zeros_like(x)

    if random_start:
        delta = delta.uniform_(-eps, eps)

    x_adv = torch.clamp(x + delta, 0, 1)
    x_adv.requires_grad = True

    optim = torch.optim.Adam([x_adv], betas=betas, maximize=not descent, lr=lr)
    scaler = torch.amp.GradScaler()

    for _ in range(steps):
        optim.zero_grad()

        y = model(x_adv).sum()

        # fp32 version:
        # y.backward()
        # optim.step()

        # fp16 version:
        scaler.scale(y).backward()
        scaler.step(optim)
        scaler.update()

        x_adv.data.clamp_(0, 1)

    return x_adv.clone().detach()


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    norm: str,
    eps: float,
    step_size: float,
    steps: int,
    adv_targets: torch.Tensor = None,
    attack_labels: torch.LongTensor = None,
    random_start: bool = False,
    descent: bool = False,  # descent minimizes loss
    logsumexp: bool = False,
) -> torch.Tensor:
    """Perform PGD attack or SGLD sampling."""
    assert not model.training
    assert not x.requires_grad
    if steps == 0:
        return x.clone()
    x0 = x.clone().detach()
    step_class = L2Step if norm == "L2" else LinfStep
    step = step_class(eps=eps, orig_input=x0, step_size=step_size)
    if random_start:
        x = step.random_perturb(x)

    for i in range(steps):
        x = x.clone().detach().requires_grad_(True)

        if logsumexp:
            # Using log-sum-exp as energy function
            logits = model(x, y=None)
            loss = torch.logsumexp(logits, 1).sum()
        else:
            if attack_labels is None:
                logits = model(x)
            else:
                logits = model(x, y=attack_labels)
            if adv_targets is None:
                loss = logits.sum()
            else:
                # since by default we maximize loss, we want the lowest cross entropy
                loss = -F.cross_entropy(logits, adv_targets)

        if descent:
            loss = -loss

        (grad,) = torch.autograd.grad(
            outputs=loss,
            inputs=[x],
            grad_outputs=None,
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
            allow_unused=False,
        )

        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)

    return x.clone().detach()


def pgd_attack_xent(
    model: nn.Module,
    x: torch.Tensor,
    true_labels: torch.LongTensor,
    norm: str,
    eps: float,
    step_size: float,
    steps: int,
    random_start: bool = False,
) -> torch.Tensor:
    """
    Perform untargeted PGD attack using cross-entropy loss.

    Args:
        model: Model to attack
        x: Clean inputs
        true_labels: True labels for the inputs
        norm: Norm to use for the attack ("L2" or "Linf")
        eps: Maximum perturbation size
        step_size: Step size for each iteration
        steps: Number of attack iterations
        random_start: Whether to start with random perturbation

    Returns:
        Adversarial examples that maximize the loss for the true labels
    """
    # Standard adversarial training uses pgd attack with model in training mode
    # assert not model.training
    assert not x.requires_grad

    if steps == 0:
        return x.clone()

    x0 = x.clone().detach()
    step_class = L2Step if norm == "L2" else LinfStep
    step = step_class(eps=eps, orig_input=x0, step_size=step_size)

    if random_start:
        x = step.random_perturb(x)

    for _ in range(steps):
        x = x.clone().detach().requires_grad_(True)
        logits = model(x, y=None)

        # Maximize loss for true label (minimize negative loss)
        loss = F.cross_entropy(logits, true_labels)

        (grad,) = torch.autograd.grad(
            outputs=loss,
            inputs=[x],
            grad_outputs=None,
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
            allow_unused=False,
        )
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
    return x.clone().detach()

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
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
    logsumexp: bool = False
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

    # SGLD parameters with improved settings
    initial_sgld_step_size = step_size  # 0.2
    temperature = 1.0  # Increased for better exploration (was 0.1)

    # Gentler decay schedule
    sgld_step_sizes = []
    decay_rate = 0.05  # Slower decay (was 0.1)
    power = 0.5      # Less aggressive power (was 0.75)
    min_step_size = 0.02  # Slightly higher minimum step size (was 0.01)

    # Pre-compute step sizes
    for t in range(steps):
        # Calculate step size using gentler polynomial decay
        current_step = initial_sgld_step_size / ((1 + decay_rate * t) ** power)
        # Ensure minimum step size
        current_step = max(current_step, min_step_size)
        sgld_step_sizes.append(current_step)

    # Initialize momentum to zero
    momentum = None

    for i in range(steps):
        x = x.clone().detach().requires_grad_(True)

        if logsumexp:
            # Using log-sum-exp as energy function
            logits = model(x, y=None)
            loss = torch.logsumexp(logits, 1).sum()

            # logits = model(x, y=None)
            # num_classes = logits.size(1)
            # uniform_targets = torch.full((logits.size(0), num_classes), 1.0 / num_classes).to(x.device)
            # loss = F.kl_div(F.log_softmax(logits, dim=1), uniform_targets, reduction='none').sum(dim=1).sum()
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

        # with torch.no_grad():
        #     if logsumexp:
        #         # Get the pre-computed step size for this iteration
        #         current_step_size = sgld_step_sizes[i]

        #         # Update momentum (initialize if first iteration)
        #         if momentum is None:
        #             momentum = torch.zeros_like(grad)

        #         # Momentum update with beta=0.9
        #         beta = 0.9
        #         momentum = beta * momentum + (1 - beta) * grad

        #         # Adjust noise scale based on current step size
        #         noise_scale = torch.sqrt(torch.tensor(2.0 * temperature * current_step_size, device=x.device))
        #         noise = torch.randn_like(x) * noise_scale

        #         # Apply SGLD update with momentum
        #         # x = x + current_step_size * momentum + noise
        #         x = x + current_step_size * momentum

        #         # Project back to ensure we stay within bounds
        #         x = step.project(x)

        #         # # Optional: log step size every 5 steps
        #         # if i % 5 == 0:
        #         #     print(f"Step {i}, step size: {current_step_size:.5f}, noise scale: {noise_scale.item():.5f}")
        #     else:
        #         # Standard PGD update
        #         x = step.step(x, grad)
        #         x = step.project(x)

    # Remove debug breakpoint for production
    # import ipdb; ipdb.set_trace()
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


def adam_pgd_l2(
    model: nn.Module,
    x: torch.Tensor,
    step_size=0.01,
    eps=0.01,
    num_steps=50,
    clamp=(0, 1),
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Performs an L2 PGD attack with updates: grad / ||grad|| * step_size.

    Args:
        model (nn.Module): The target model to attack.
        x (torch.Tensor): The input tensor (batch of images).
        step_size (float): Step size for each PGD step.
        epsilon (float): Maximum perturbation budget (L2 norm of perturbation).
        num_steps (int): Number of PGD attack iterations.
        clamp (tuple): Tuple (min, max) for clamping pixel values.
        device (str): Computation device.

    Returns:
        torch.Tensor: Adversarial examples.
    """
    model.eval()

    x = x.to(device)
    ys = []

    delta = torch.zeros_like(x).uniform_(-eps, eps).requires_grad_(True)
    delta = delta.to(device)
    delta.requires_grad = True

    for _ in range(num_steps):
        if delta.grad is not None:
            delta.grad.detach_()
            delta.grad.zero_()

        x_adv = x + delta
        x_adv = torch.clamp(
            x_adv, clamp[0], clamp[1]
        )  # Ensure valid image range

        outputs = model(x_adv)
        loss = outputs
        ys.append(float(loss))

        loss.backward()

        grad = delta.grad  # Shape: (B, C, H, W)
        grad_norm = (
            grad.view(grad.size(0), -1)
            .norm(p=2, dim=1, keepdim=True)
            .view(-1, 1, 1, 1)
        )
        scaled_grad = (
            grad / (grad_norm + 1e-10) * step_size
        )  # Avoid divide-by-zero

        delta = delta + scaled_grad

        delta = delta.detach()
        delta.requires_grad = True

    x_adv = torch.clamp(
        x + delta, clamp[0], clamp[1]
    )  # Clamp to valid image range
    return x_adv.clone().detach(), ys


def adam_gd(
    model: nn.Module,
    x: torch.Tensor,
    eps: float = 0.01,
    num_steps: int = 50,  # default is for generation
    lr: float | None = 1e-3,
    betas: tuple = (0.9, 0.999),
    random_start: bool = False,
    descent: bool = False,
) -> torch.Tensor:
    model.eval()
    assert not model.training
    assert not x.requires_grad

    if num_steps == 0:
        return x

    delta = torch.zeros_like(x)

    if random_start:
        delta = delta.uniform_(-eps, eps)

    x_adv = x + delta
    x_adv.requires_grad = True

    optimizer = torch.optim.Adam([x_adv], betas=betas, maximize=True, lr=lr)
    ys = []
    norms = []

    for _ in range(num_steps):
        optimizer.zero_grad()

        y = model(x_adv)
        ys.append(float(y))
        norms.append(torch.norm(x_adv - x).item())

        y.backward()
        optimizer.step()
        x_adv.data.clamp_(0, 1)

    return x_adv.clone().detach(), ys, norms


def diff(img1, img2):
    """
    Displays a heatmap showing differences between two PyTorch tensors.
    Assumes images are in the range [0, 1].
    """
    dif = (img1 - img2).squeeze(0)
    dif_normalized = dif / (torch.max(torch.abs(dif)) + 1e-10)
    dif_scaled = (dif_normalized + 1) / 2  # Shift the range to [0, 1]
    dif_np = dif_scaled.detach().cpu().numpy()

    if img1.dim() == 3:  # RGB image
        dif_np = dif_np.transpose(1, 2, 0)

    dif_norm = torch.norm(dif).item()

    # display a plot with 3 side by side images: img1, img2, and img2-img1
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img1.cpu().numpy().transpose(1, 2, 0), interpolation="none")
    plt.axis("off")
    plt.title("Original")
    plt.subplot(1, 3, 2)
    plt.imshow(img2.cpu().numpy().transpose(1, 2, 0), interpolation="none")
    plt.axis("off")
    plt.title("Adversarial")
    plt.subplot(1, 3, 3)
    plt.imshow(dif_np, interpolation="none")
    plt.axis("off")
    plt.title(f"Diff. (L2={dif_norm:.2f})")
    plt.show()
    return dif_np

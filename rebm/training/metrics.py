"""Metrics computation and tracking for training and evaluation."""

import dataclasses
from typing import TYPE_CHECKING, Callable

import einops
import torch
from torch import nn

import rebm.training.misc
from rebm.attacks.adv_attacks import pgd_attack, pgd_attack_xent
from rebm.eval.eval_utils import (
    compute_img_diff,
    generate_outdist_adv_images,
    get_auc,
)


def assert_no_grad(model):
    """
    Assert that all gradients in the model are either None or zeros.

    Args:
        model: PyTorch model to check

    Raises:
        AssertionError: If any parameter gradients are neither None nor zero
    """
    for p in model.parameters():
        if p.grad is not None and not torch.all(p.grad.eq(0)):
            raise AssertionError(
                "Some parameter gradients are neither None nor zero."
            )

    return True


if TYPE_CHECKING:
    from rebm.training.train import TrainConfig


@dataclasses.dataclass
class EBMMetrics:
    # AUC metrics
    clean_auc: float
    adv_auc: float

    # Loss metrics
    ebm_binary_loss: torch.Tensor | None = None
    ebm_total_loss: torch.Tensor | None = None
    r1: torch.Tensor | None = None

    # Image tensors
    indist_imgs: torch.Tensor | None = None
    outdist_imgs_clean: torch.Tensor | None = None
    adv_imgs: torch.Tensor | None = None
    outdist_imgs_error: torch.Tensor | None = None

    # Classifier logits
    clf_indist: torch.Tensor | None = None
    clf_outdist: torch.Tensor | None = None
    clf_adv: torch.Tensor | None = None

    # Distance metrics
    l2_dist_relative: float | None = None

    def to_simple_dict(self) -> dict[str, float]:
        return {
            field.name: (val.item() if torch.is_tensor(val) else val)
            for field in dataclasses.fields(self)
            if (val := getattr(self, field.name)) is not None
            and not (torch.is_tensor(val) and val.numel() > 1)
        }


@dataclasses.dataclass
class ImageGenerationMetrics:
    """Class to track image generation metrics (FID scores)."""

    fid: float | None = None
    best_fid: float = float("inf")

    def update(self, fid: float | None) -> bool:
        """Update FID metrics and check if new best.

        Args:
            fid: Current FID score

        Returns:
            True if this is a new best FID
        """
        self.fid = fid

        # Check if we have a new best FID
        new_best = False
        if fid is not None and fid < self.best_fid:
            self.best_fid = fid
            new_best = True

        return new_best


@dataclasses.dataclass
class ClassificationMetrics:
    """Class to track classification metrics."""

    test_acc: float | None = None
    robust_test_acc: float | None = None
    best_robust_test_acc: float | None = None

    def update(
        self,
        *,  # Force keyword arguments
        test_acc: float,
        robust_test_acc: float | None = None,
    ) -> bool:
        self.test_acc = test_acc

        # Update robust accuracy metrics
        if robust_test_acc is not None:
            self.robust_test_acc = robust_test_acc

        # Check if we have a new best robust test accuracy
        new_best = False
        if robust_test_acc is not None and (
            self.best_robust_test_acc is None
            or robust_test_acc > self.best_robust_test_acc
        ):
            self.best_robust_test_acc = robust_test_acc
            new_best = True

        return new_best


def compute_auc_metrics(
    model,
    indist_imgs,
    indist_labels,
    adv_imgs,
    outdist_imgs,
    attack_labels,
) -> tuple[float, float]:
    """Compute AUC metrics with no gradient tracking.

    Returns:
        tuple[float, float]: (adv_auc, clean_auc)
    """
    assert_no_grad(model)
    assert not model.training

    with torch.no_grad():
        inputs = torch.cat([indist_imgs, adv_imgs, outdist_imgs])
        labels = torch.cat([indist_labels, attack_labels, attack_labels])

        batch_logits = model(inputs, labels)
        indist_logits, adv_logits, outdist_logits = torch.chunk(batch_logits, 3)

        adv_auc = get_auc(
            pos=indist_logits.cpu().numpy(), neg=adv_logits.cpu().numpy()
        )
        clean_auc = get_auc(
            pos=indist_logits.cpu().numpy(),
            neg=outdist_logits.cpu().numpy(),
        )

    # Ensure no gradients were accumulated
    assert_no_grad(model)
    return adv_auc, clean_auc


def compute_ebm_metrics(
    *,
    indist_imgs: torch.Tensor,
    indist_labels: torch.Tensor,
    outdist_imgs: torch.Tensor,
    outdist_step: int,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    model: nn.Module,
    cfg: "TrainConfig",
) -> EBMMetrics:
    """Compute metrics during training including loss, AUC, and R1 regularization."""
    model.eval()
    adv_imgs, attack_labels = generate_outdist_adv_images(
        model, outdist_imgs, cfg, outdist_step, indist_labels=indist_labels
    )

    if cfg.indist_perturb:
        indist_imgs = pgd_attack(
            model,
            indist_imgs,
            norm="L2",
            eps=cfg.indist_perturb_eps,
            step_size=cfg.attack.step_size,
            steps=cfg.indist_perturb_steps,
            attack_labels=indist_labels,
            descent=True,
        )

    l2_dist = (
        torch.norm(
            einops.rearrange(adv_imgs - outdist_imgs, "b ... -> b (...)"), dim=1
        )
        .mean()
        .item()
    )
    l2_dist_relative = (
        1
        if outdist_step == 0
        else l2_dist / (outdist_step * cfg.attack.step_size)
    )

    adv_auc, clean_auc = compute_auc_metrics(
        model,
        indist_imgs,
        indist_labels,
        adv_imgs,
        outdist_imgs,
        attack_labels,
    )

    indist_target = torch.ones(indist_imgs.shape[0]).to(indist_imgs.device)
    adv_target = torch.zeros(indist_imgs.shape[0]).to(indist_imgs.device)

    model.train()
    # Compute R1 only if using it in loss (r1reg > 0) or tracking it (track_r1=True)
    if cfg.r1reg > 0 or cfg.track_r1:
        indist_imgs.requires_grad_()
        if cfg.logsumexp:
            indist_logits = torch.logsumexp(model(x=indist_imgs, y=None), dim=1)
        else:
            indist_logits = model(x=indist_imgs, y=indist_labels)
        r1 = rebm.training.misc.r1_reg(indist_logits, indist_imgs)
    else:
        r1 = 0
        if cfg.logsumexp:
            indist_logits = torch.logsumexp(model(x=indist_imgs, y=None), dim=1)
        else:
            indist_logits = model(x=indist_imgs, y=indist_labels)

    # Use out-of-distribution adversarial examples
    if cfg.logsumexp:
        adv_logits = torch.logsumexp(model(adv_imgs, y=None), dim=1)
    else:
        adv_logits = model(adv_imgs, attack_labels)
    logits = torch.cat([indist_logits, adv_logits])
    targets = torch.cat([indist_target, adv_target])

    ebm_binary_loss = criterion(logits, targets)
    ebm_total_loss = ebm_binary_loss + cfg.r1reg * r1

    ret_metrics_dict = dict(
        ebm_total_loss=ebm_total_loss,
        ebm_binary_loss=ebm_binary_loss.detach().item(),
        r1=r1.detach().item() if isinstance(r1, torch.Tensor) else r1,
        l2_dist_relative=l2_dist_relative,
        indist_imgs=indist_imgs.detach(),
        outdist_imgs_clean=outdist_imgs,
        adv_imgs=adv_imgs.detach(),
        outdist_imgs_error=compute_img_diff(adv_imgs, outdist_imgs).detach(),
        adv_auc=adv_auc,
        clean_auc=clean_auc,
    )

    return EBMMetrics(**ret_metrics_dict)


def compute_clf_adv_loss(
    *,
    indist_imgs: torch.Tensor,
    indist_labels: torch.Tensor,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    model: nn.Module,
    cfg: "TrainConfig",
) -> torch.Tensor:
    """Compute classification loss on adversarial in-distribution examples."""
    assert (
        cfg.indist_attack_clf.max_steps
        == cfg.indist_attack_clf.fixed_steps
        == cfg.indist_attack_clf.start_step
    )
    indist_adv_imgs = pgd_attack_xent(
        model,
        indist_imgs,
        indist_labels,
        norm=cfg.indist_attack_clf.norm,
        eps=cfg.indist_attack_clf.eps,
        step_size=cfg.indist_attack_clf.step_size,
        steps=cfg.indist_attack_clf.max_steps,
    )
    indist_adv_logits = model(indist_adv_imgs)

    # Compute loss from adversarial samples
    loss = criterion(indist_adv_logits, indist_labels)

    return loss

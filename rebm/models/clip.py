import pathlib

import clip
import einops
import numpy as np
import torch
import transformers
from jaxtyping import Float
from scipy import stats
from torch import nn

# import torch.nn.attention
from torchvision import transforms


def _clip_normalize(imgs: Float[torch.Tensor, "b c h w"]):
    """
    Taken from https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py#L85.
    """
    return transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )(imgs)


class OneClassCLIP(nn.Module):
    """A CLIP model just outputs a single score for the given class."""

    def __init__(
        self,
        class_text: str,
        scores_on_all_cats_path: pathlib.Path | None = None,
    ):
        super().__init__()

        # TODO: Specify which clip model to load
        self.clip_model, _ = clip.load("ViT-B/32")
        self.img_size = 224

        self.text = class_text
        self.text_embeddings = clip.tokenize([class_text]).to(self.device)

        if scores_on_all_cats_path is not None:
            self.scores_on_all_cats = np.load(scores_on_all_cats_path)
        else:
            self.scores_on_all_cats = None

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, imgs: Float[torch.Tensor, "b c h w"]):
        """imgs should have pixels in the range [0, 1]."""
        resized_imgs = nn.functional.interpolate(
            imgs,
            size=(self.img_size, self.img_size),
            mode="bicubic",
        )
        normalized_imgs = _clip_normalize(resized_imgs)

        logits_per_image, _ = self.clip_model(
            normalized_imgs, self.text_embeddings
        )
        return einops.rearrange(logits_per_image, "b 1 -> b")

    def get_name(self):
        return "OneClassCLIP"

    # below functions turn scores into percentiles and vise versa, found from running on the complete 5k cats dataset
    def get_score_from_percentile(self, percentile):
        return np.percentile(self.scores_on_all_cats, percentile * 100)

    def get_percentile_from_score(self, score):
        return stats.percentileofscore(self.scores_on_all_cats, score) / 100


class OneClassHFCLIP(nn.Module):
    """A CLIP model just outputs a single score for the given class."""

    def __init__(
        self,
        class_text: str,
        model_id="openai/clip-vit-base-patch32",
        **hf_kwargs,
    ):
        super().__init__()

        self.clip_model: transformers.CLIPModel = (
            transformers.CLIPModel.from_pretrained(model_id, **hf_kwargs)
        )
        self.bias = nn.Parameter(torch.ones(1, dtype=torch.float32) * -25)
        self.scale = nn.Parameter(torch.ones(1, dtype=torch.float32) * 0.3)
        self.class_text = class_text

        # Make the text model not trainable
        for param in self.clip_model.text_model.parameters():
            param.requires_grad = False
        for param in self.clip_model.text_projection.parameters():
            param.requires_grad = False

        self.tokenizer: transformers.CLIPTokenizer = (
            transformers.CLIPTokenizer.from_pretrained(model_id)
        )
        self.img_size = self.clip_model.config.vision_config.image_size

    def forward(self, imgs: Float[torch.Tensor, "b c h w"]) -> torch.Tensor:
        resized_imgs = nn.functional.interpolate(
            imgs,
            size=(self.img_size, self.img_size),
            mode="bicubic",
        )
        normalized_imgs = _clip_normalize(resized_imgs)

        # See https://github.com/pytorch/pytorch/issues/116350#issuecomment-1954667011
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            clip_model_output = self.clip_model.forward(
                pixel_values=normalized_imgs,
                input_ids=torch.tensor(
                    self.tokenizer([self.class_text]).input_ids,
                    device=self.device,
                ),
                return_loss=False,
                output_attentions=False,
                output_hidden_states=False,
            )
        return (
            einops.rearrange(clip_model_output.logits_per_image, "b 1 -> b")
            + self.bias
        ) * self.scale

    @property
    def device(self):
        return next(self.parameters()).device

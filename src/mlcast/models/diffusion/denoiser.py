"""Diffusion-step-aware denoising network for latent diffusion."""

import math

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from loguru import logger


class DiffusionStepEmbedding(nn.Module):
    """Sinusoidal embedding of the diffusion step followed by a small MLP.

    The diffusion step is the position in the noising/denoising chain, not a
    forecast time. The diffusion literature (and LDCast) usually calls this a
    "timestep embedding".

    Parameters
    ----------
    embedding_dim : int
        Number of channels in the generated diffusion step embedding.
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, diffusion_step_index: Int[torch.Tensor, " batch"]) -> Float[torch.Tensor, "batch embedding_dim"]:
        """Embed integer diffusion step indices.

        Parameters
        ----------
        diffusion_step_index : Int[torch.Tensor, "batch"]
            Index into the diffusion schedule for each sample.

        Returns
        -------
        Float[torch.Tensor, "batch embedding_dim"]
            Projected sinusoidal diffusion step embeddings.
        """
        half_dim = self.embedding_dim // 2
        frequencies = torch.exp(
            torch.arange(half_dim, device=diffusion_step_index.device, dtype=torch.float32)
            * -(math.log(10_000.0) / max(half_dim - 1, 1))
        )
        args = diffusion_step_index.float()[:, None] * frequencies[None]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if embedding.shape[-1] < self.embedding_dim:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return self.projection(embedding)


class _DenoiserBlock(nn.Module):
    """Internal residual denoising block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    diffusion_step_channels : int
        Number of channels in the diffusion step embedding.
    """

    def __init__(self, in_channels: int, out_channels: int, diffusion_step_channels: int) -> None:
        super().__init__()
        self.diffusion_step_projection = nn.Linear(diffusion_step_channels, out_channels)
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip_connection = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "batch in_channels time height width"],
        diffusion_step_embedding: Float[torch.Tensor, "batch embedding_dim"],
    ) -> Float[torch.Tensor, "batch out_channels time height width"]:
        """Apply diffusion-step-conditioned residual denoising.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch in_channels time height width"]
            Hidden denoising tensor.
        diffusion_step_embedding : Float[torch.Tensor, "batch embedding_dim"]
            Diffusion step embedding for each batch item.

        Returns
        -------
        Float[torch.Tensor, "batch out_channels time height width"]
            Updated hidden tensor.
        """
        diffusion_step_bias = self.diffusion_step_projection(diffusion_step_embedding)[:, :, None, None, None]
        h = self.net(x)
        return self.skip_connection(x) + h + diffusion_step_bias


class _SpatialDownsample(nn.Module):
    """Halve latent spatial resolution while preserving time."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv3d(channels, channels, kernel_size=3, stride=(1, 2, 2), padding=1)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch channels time height width"]
    ) -> Float[torch.Tensor, "batch channels time out_height out_width"]:
        """Downsample the spatial dimensions of a latent tensor.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch channels time height width"]
            Channel-first latent tensor.

        Returns
        -------
        Float[torch.Tensor, "batch channels time out_height out_width"]
            Tensor with half spatial resolution.
        """
        return self.op(x)


class _SpatialUpsample(nn.Module):
    """Double latent spatial resolution while preserving time."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.ConvTranspose3d(
            channels,
            channels,
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1,
            output_padding=(0, 1, 1),
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch channels time height width"]
    ) -> Float[torch.Tensor, "batch channels time out_height out_width"]:
        """Upsample the spatial dimensions of a latent tensor.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch channels time height width"]
            Channel-first latent tensor.

        Returns
        -------
        Float[torch.Tensor, "batch channels time out_height out_width"]
            Tensor with doubled spatial resolution.
        """
        return self.op(x)


class DenoiserUNet(nn.Module):
    """Compact diffusion-step-aware U-Net denoiser for latent tensors.

    This is a real U-Net-style architecture: it builds a spatial downsampling
    path, applies a bottleneck at the lowest spatial resolution, upsamples back
    to the original latent resolution, and concatenates matching-resolution
    skip connections from the down path into the up path. It differs from a
    plain image U-Net because it operates on 3D latent tensors and only changes
    spatial resolution; the temporal dimension is preserved throughout. Each
    residual block also receives a diffusion step embedding.

    Parameters
    ----------
    latent_channels : int
        Number of channels in the noisy target latent.
    condition_channels : int
        Number of channels emitted by the conditioner.
    hidden_channels : int, optional
        Number of hidden channels in the denoiser. Default is ``32``.
    num_blocks : int, optional
        Number of U-Net resolution levels. Default is ``2``.
    """

    def __init__(
        self,
        latent_channels: int,
        condition_channels: int,
        hidden_channels: int = 32,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks ({num_blocks}) must be at least 1.")

        self.latent_channels = latent_channels
        self.condition_channels = condition_channels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.diffusion_step_embedding = DiffusionStepEmbedding(hidden_channels)
        self.input_projection = nn.Conv3d(latent_channels + condition_channels, hidden_channels, kernel_size=1)
        self.down_blocks = nn.ModuleList(
            _DenoiserBlock(hidden_channels, hidden_channels, hidden_channels) for _ in range(num_blocks)
        )
        self.downsamples = nn.ModuleList(_SpatialDownsample(hidden_channels) for _ in range(num_blocks - 1))
        self.bottleneck = _DenoiserBlock(hidden_channels, hidden_channels, hidden_channels)
        self.upsamples = nn.ModuleList(_SpatialUpsample(hidden_channels) for _ in range(num_blocks - 1))
        self.up_blocks = nn.ModuleList(
            _DenoiserBlock(hidden_channels * 2, hidden_channels, hidden_channels) for _ in range(num_blocks - 1)
        )
        self.output_projection = nn.Conv3d(hidden_channels, latent_channels, kernel_size=1)
        self._warned_context_resize = False

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        noisy: Float[torch.Tensor, "batch latent_channels forecast_time height width"],
        diffusion_step_index: Int[torch.Tensor, " batch"],
        context: Float[torch.Tensor, "batch condition_channels input_time height width"],
    ) -> Float[torch.Tensor, "batch latent_channels forecast_time height width"]:
        """Predict noise in a noised latent target.

        Parameters
        ----------
        noisy : Float[torch.Tensor, "batch latent_channels forecast_time height width"]
            Noised target latent.
        diffusion_step_index : Int[torch.Tensor, "batch"]
            Index into the diffusion schedule for each sample.
        context : Float[torch.Tensor, "batch condition_channels input_time height width"]
            Conditioning context from the input-history latent.

        Returns
        -------
        Float[torch.Tensor, "batch latent_channels forecast_time height width"]
            Predicted noise tensor.
        """
        if context.shape[2] != noisy.shape[2]:
            # Warn once per instance rather than on every denoising step.
            if not self._warned_context_resize:
                logger.warning(
                    "DenoiserUNet: conditioning context has {} time steps but the noisy target has {}; "
                    "resizing the context to {} with nearest-neighbour interpolation.",
                    context.shape[2],
                    noisy.shape[2],
                    tuple(noisy.shape[2:]),
                )
                self._warned_context_resize = True
            context = torch.nn.functional.interpolate(context, size=noisy.shape[2:], mode="nearest")

        x = self.input_projection(torch.cat([noisy, context], dim=1))
        diffusion_step_embedding = self.diffusion_step_embedding(diffusion_step_index)

        skips: list[torch.Tensor] = []
        for block_idx, block in enumerate(self.down_blocks):
            x = block(x, diffusion_step_embedding)
            if block_idx < len(self.downsamples):
                skips.append(x)
                x = self.downsamples[block_idx](x)

        x = self.bottleneck(x, diffusion_step_embedding)

        for upsample, block in zip(self.upsamples, self.up_blocks, strict=True):
            x = upsample(x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = block(torch.cat([x, skip], dim=1), diffusion_step_embedding)

        return self.output_projection(x)

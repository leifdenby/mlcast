"""Diffusion noise schedules."""

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped


class DiffusionScheduler:
    """Linear-beta diffusion scheduler.

    Parameters
    ----------
    n_diffusion_steps : int, optional
        Number of steps in the diffusion (noising/denoising) chain. Default is ``100``.
    beta_start : float, optional
        Initial beta value. Default is ``1e-4``.
    beta_end : float, optional
        Final beta value. Default is ``2e-2``.

    Raises
    ------
    ValueError
        If ``n_diffusion_steps`` is less than 1.
    """

    def __init__(self, n_diffusion_steps: int = 100, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        if n_diffusion_steps < 1:
            raise ValueError(f"n_diffusion_steps ({n_diffusion_steps}) must be at least 1.")
        self.n_diffusion_steps = n_diffusion_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

    def buffers(
        self, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> dict[str, Float[torch.Tensor, " n_diffusion_steps"]]:
        """Build schedule tensors for registration as module buffers.

        Parameters
        ----------
        device : torch.device
            Device on which buffers should be allocated.
        dtype : torch.dtype, optional
            Floating-point dtype for schedule tensors. Default is
            ``torch.float32``.

        Returns
        -------
        dict of str to Float[torch.Tensor, "n_diffusion_steps"]
            Schedule tensors used for forward and reverse diffusion.
        """
        betas = torch.linspace(self.beta_start, self.beta_end, self.n_diffusion_steps, device=device, dtype=dtype)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        }


@jaxtyped(typechecker=beartype)
def extract_schedule_value(
    values: Float[torch.Tensor, " n_diffusion_steps"],
    diffusion_step_index: Int[torch.Tensor, " batch"],
    target_shape: torch.Size,
) -> Float[torch.Tensor, " batch *broadcast_dims"]:
    """Gather schedule values and reshape them for broadcasting.

    Parameters
    ----------
    values : Float[torch.Tensor, "n_diffusion_steps"]
        One-dimensional schedule tensor.
    diffusion_step_index : Int[torch.Tensor, "batch"]
        Index into the diffusion schedule for each batch item.
    target_shape : torch.Size
        Shape of the target tensor the values should broadcast against.

    Returns
    -------
    Float[torch.Tensor, "batch *broadcast_dims"]
        Gathered values reshaped to ``(batch, 1, ..., 1)``.
    """
    gathered = values.gather(0, diffusion_step_index)
    return gathered.reshape(diffusion_step_index.shape[0], *([1] * (len(target_shape) - 1)))

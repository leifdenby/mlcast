import pytest
import torch

from mlcast.losses import AFCRPS, CRPS, MaskedLoss, build_loss


def test_build_loss_invalid_type():
    """Verify build_loss raises TypeError if loss_class is neither a string nor a class."""
    with pytest.raises(TypeError, match="loss_class must be a string or a class"):
        build_loss(loss_class=123)

    with pytest.raises(TypeError, match="loss_class must be a string or a class"):
        build_loss(loss_class=None)


def test_build_loss_invalid_string():
    """Verify build_loss raises ValueError for unknown string names."""
    with pytest.raises(ValueError, match="Unknown loss class 'unknown'"):
        build_loss(loss_class="unknown")


@pytest.mark.parametrize("loss_class", [CRPS, AFCRPS])
def test_crps_loss_shapes(loss_class):
    """Verify that CRPS and afCRPS losses output scalar for mean reduction,
    and correct shapes for other reductions."""
    # (B, T, M, *D) - M is ensemble size
    B, T, M, H, W = 2, 4, 3, 16, 16
    preds = torch.randn(B, T, M, H, W)
    # (B, T, C, *D) - C is usually 1
    target = torch.randn(B, T, 1, H, W)

    # mean reduction (default)
    loss_fn = loss_class(reduction="mean")
    loss = loss_fn(preds, target)
    assert loss.ndim == 0, f"{loss_class.__name__} with mean reduction should be scalar"

    # sum reduction
    loss_fn = loss_class(reduction="sum")
    loss = loss_fn(preds, target)
    assert loss.ndim == 0, f"{loss_class.__name__} with sum reduction should be scalar"

    # none reduction (elementwise)
    loss_fn = loss_class(reduction="none")
    loss = loss_fn(preds, target)
    # Expected output for none reduction is (B, T, 1, H, W)
    assert loss.shape == (B, T, 1, H, W), f"{loss_class.__name__} none reduction shape mismatch"

    # test with temporal_lambda > 0
    loss_fn_temporal = loss_class(reduction="none", temporal_lambda=0.1)
    loss_temporal = loss_fn_temporal(preds, target)
    # Temporal penalty is broadcast across all T steps — output shape must
    # preserve the full T dimension so MaskedLoss can apply the mask correctly.
    assert loss_temporal.shape == (B, T, 1, H, W), f"{loss_class.__name__} temporal shape mismatch"


@pytest.mark.parametrize("loss_class", [CRPS, AFCRPS])
def test_masked_loss_with_temporal_lambda(loss_class: type) -> None:
    """MaskedLoss wrapping CRPS/AFCRPS with temporal_lambda > 0 must produce a
    non-zero loss when the mask has valid pixels.

    Regression test for the shape-mismatch bug where temporal_lambda caused the
    T dimension to collapse, making broadcast_factor = 0 and valid_pixels = 0
    even for fully valid batches.
    """
    B, T, M, H, W = 2, 4, 3, 16, 16
    preds = torch.randn(B, T, M, H, W)
    target = torch.randn(B, T, 1, H, W)
    mask = torch.ones(B, T, 1, H, W)  # all valid

    criterion = MaskedLoss(loss_class(temporal_lambda=0.1, reduction="none"))
    loss = criterion(preds, target, mask)

    assert loss.isfinite(), f"{loss_class.__name__}: loss should be finite with valid mask"
    assert loss.item() > 0, f"{loss_class.__name__}: loss should be > 0 with valid mask"

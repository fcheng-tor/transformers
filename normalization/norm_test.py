from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _maybe_call(mod, fn_name: str, *args, **kwargs):
    fn = getattr(mod, fn_name, None)
    if fn is None:
        pytest.skip(
            f"`{mod.__name__}.{fn_name}` not implemented yet. "
            "Implement it and the test will start exercising it."
        )
    return fn(*args, **kwargs)


def _assert_allclose(a: torch.Tensor, b: torch.Tensor, *, rtol=1e-4, atol=1e-5):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_layer_norm_forward_backward_matches_torch(dtype):
    """
    Skeleton for LayerNorm on last dimension.

    PyTorch reference: torch.nn.functional.layer_norm

    Your implementation (expected signatures; adjust if you prefer different API):
      - y, cache = layer_norm_forward(x, gamma, beta, eps=1e-5)
      - dx, dgamma, dbeta = layer_norm_backward(dy, cache)
    """
    torch.manual_seed(0)

    N, T, D = 2, 3, 8
    device = torch.device("cpu")
    x = torch.randn(N, T, D, dtype=dtype, device=device, requires_grad=True)
    gamma = torch.randn(D, dtype=dtype, device=device, requires_grad=True)
    beta = torch.randn(D, dtype=dtype, device=device, requires_grad=True)
    dy = torch.randn(N, T, D, dtype=dtype)
    eps = 1e-5

    y_ref = torch.nn.functional.layer_norm(
        x, normalized_shape=(D,), weight=gamma, bias=beta, eps=eps
    )
    loss_ref = (y_ref * dy).sum()
    loss_ref.backward()
    dx_ref = x.grad.detach()
    dgamma_ref = gamma.grad.detach()
    dbeta_ref = beta.grad.detach()

    layer_norm = importlib.import_module("normalization.layer_norm")
    y, cache = _maybe_call(layer_norm, "layer_norm_forward", x.detach(), gamma.detach(), beta.detach(), eps=eps)
    dx, dgamma, dbeta = _maybe_call(layer_norm, "layer_norm_backward", dy, cache)

    _assert_allclose(y, y_ref.detach())
    _assert_allclose(dx, dx_ref)
    _assert_allclose(dgamma, dgamma_ref)
    _assert_allclose(dbeta, dbeta_ref)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_instance_norm_forward_backward_matches_torch(dtype):
    """
    Skeleton for InstanceNorm over spatial dims (per-sample, per-channel).

    PyTorch reference: torch.nn.functional.instance_norm

    Your implementation (expected signatures):
      - y, cache = instance_norm_forward(x, gamma, beta, eps=1e-5)
      - dx, dgamma, dbeta = instance_norm_backward(dy, cache)

    Shapes:
      x: (N, C, H, W)
      gamma/beta: (C,)
    """
    torch.manual_seed(0)

    N, C, H, W = 2, 3, 4, 5
    device = torch.device("cpu")
    x = torch.randn(N, C, H, W, dtype=dtype, device=device, requires_grad=True)
    gamma = torch.randn(C, dtype=dtype, device=device, requires_grad=True)
    beta = torch.randn(C, dtype=dtype, device=device, requires_grad=True)
    dy = torch.randn(N, C, H, W, dtype=dtype)
    eps = 1e-5

    # For reference op, provide running stats=None to use per-instance stats.
    y_ref = torch.nn.functional.instance_norm(
        x,
        running_mean=None,
        running_var=None,
        weight=gamma,
        bias=beta,
        use_input_stats=True,
        momentum=0.1,
        eps=eps,
    )
    loss_ref = (y_ref * dy).sum()
    loss_ref.backward()
    dx_ref = x.grad.detach()
    dgamma_ref = gamma.grad.detach()
    dbeta_ref = beta.grad.detach()

    instance_norm = importlib.import_module("normalization.instance_norm")
    y, cache = _maybe_call(instance_norm, "instance_norm_forward", x.detach(), gamma.detach(), beta.detach(), eps=eps)
    dx, dgamma, dbeta = _maybe_call(instance_norm, "instance_norm_backward", dy, cache)

    _assert_allclose(y, y_ref.detach())
    _assert_allclose(dx, dx_ref)
    _assert_allclose(dgamma, dgamma_ref)
    _assert_allclose(dbeta, dbeta_ref)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_batch_norm_forward_backward_matches_torch(dtype):
    """
    Skeleton for BatchNorm over channel dim for NCHW inputs.

    PyTorch reference: torch.nn.functional.batch_norm in training mode

    Your implementation (expected signatures):
      - y, cache = batch_norm_forward(x, gamma, beta, eps=1e-5)
      - dx, dgamma, dbeta = batch_norm_backward(dy, cache)

    Notes:
    - Reference BN computes stats over (N, H, W) per channel for NCHW.
    """
    torch.manual_seed(0)

    N, C, H, W = 4, 3, 5, 6
    device = torch.device("cpu")
    x = torch.randn(N, C, H, W, dtype=dtype, device=device, requires_grad=True)
    gamma = torch.randn(C, dtype=dtype, device=device, requires_grad=True)
    beta = torch.randn(C, dtype=dtype, device=device, requires_grad=True)
    dy = torch.randn(N, C, H, W, dtype=dtype)
    eps = 1e-5

    # training=True via `training` arg; no running stats used for ref.
    y_ref = torch.nn.functional.batch_norm(
        x,
        running_mean=None,
        running_var=None,
        weight=gamma,
        bias=beta,
        training=True,
        momentum=0.1,
        eps=eps,
    )
    loss_ref = (y_ref * dy).sum()
    loss_ref.backward()
    dx_ref = x.grad.detach()
    dgamma_ref = gamma.grad.detach()
    dbeta_ref = beta.grad.detach()

    batch_norm = importlib.import_module("normalization.batch_norm")
    y, cache = _maybe_call(batch_norm, "batch_norm_forward", x.detach(),
                                gamma.detach(), beta.detach(), eps=eps)
    dx, dgamma, dbeta = _maybe_call(batch_norm, "batch_norm_backward", dy, cache)

    _assert_allclose(y, y_ref.detach())
    _assert_allclose(dx, dx_ref)
    _assert_allclose(dgamma, dgamma_ref)
    _assert_allclose(dbeta, dbeta_ref)


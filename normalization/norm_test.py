import pytest
import torch

import normalization.batch_norm as batch_norm
import normalization.instance_norm as instance_norm
import normalization.layer_norm as layer_norm


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
@pytest.mark.parametrize(
    "x_shape",
    [
        (8,),  # 1D tensor: normalize over the only dimension
        (4, 8),  # 2D: (batch_or_length, D)
        (2, 3, 8),  # 3D: (N, T, D) — typical sequence layout
    ],
    ids=["1d", "2d", "3d"],
)
def test_layer_norm_forward_backward_matches_torch(dtype, x_shape):
    """
    Skeleton for LayerNorm on last dimension.

    PyTorch reference: torch.nn.functional.layer_norm

    Your implementation (expected signatures; adjust if you prefer different API):
      - y, cache = layer_norm_forward(x, gamma, beta, eps=1e-5)
      - dx, dgamma, dbeta = layer_norm_backward(dy, cache)
    """
    torch.manual_seed(0)

    D = x_shape[-1]
    device = torch.device("cpu")
    x = torch.randn(x_shape, dtype=dtype, device=device, requires_grad=True)
    gamma = torch.randn(D, dtype=dtype, device=device, requires_grad=True)
    beta = torch.randn(D, dtype=dtype, device=device, requires_grad=True)
    dy = torch.randn(x_shape, dtype=dtype)
    eps = 1e-5

    y_ref = torch.nn.functional.layer_norm(
        x, normalized_shape=(D,), weight=gamma, bias=beta, eps=eps
    )
    loss_ref = (y_ref * dy).sum()
    loss_ref.backward()
    dx_ref = x.grad.detach()
    dgamma_ref = gamma.grad.detach()
    dbeta_ref = beta.grad.detach()

    y, cache = _maybe_call(layer_norm, "layer_norm_forward", x.detach(), gamma.detach(), beta.detach(), eps=eps)
    dx, dgamma, dbeta = _maybe_call(layer_norm, "layer_norm_backward", dy, cache)

    _assert_allclose(y, y_ref.detach())
    _assert_allclose(dx, dx_ref)
    _assert_allclose(dgamma, dgamma_ref)
    _assert_allclose(dbeta, dbeta_ref)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "x_shape",
    [
        (2, 3, 40),  # 3D input: 1D spatial (N, C, L)
        (2, 3, 4, 5),  # 4D: 2D spatial (N, C, H, W)
        (2, 3, 4, 5, 6),  # 5D: 3D spatial (N, C, D, H, W)
    ],
    ids=["spatial_1d", "spatial_2d", "spatial_3d"],
)
def test_instance_norm_forward_backward_matches_torch(dtype, x_shape):
    """
    Skeleton for InstanceNorm over spatial dims (per-sample, per-channel).

    PyTorch reference: torch.nn.functional.instance_norm

    Your implementation (expected signatures):
      - y, cache = instance_norm_forward(x, gamma, beta, eps=1e-5)
      - dx, dgamma, dbeta = instance_norm_backward(dy, cache)

    Shapes:
      x: (N, C, L), (N, C, H, W), or (N, C, D, H, W)
      gamma/beta: (C,)
    """
    torch.manual_seed(0)

    C = x_shape[1]
    device = torch.device("cpu")
    x = torch.randn(x_shape, dtype=dtype, device=device, requires_grad=True)
    gamma = torch.randn(C, dtype=dtype, device=device, requires_grad=True)
    beta = torch.randn(C, dtype=dtype, device=device, requires_grad=True)
    dy = torch.randn(x_shape, dtype=dtype)
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

    y, cache = _maybe_call(instance_norm, "instance_norm_forward", x.detach(), gamma.detach(), beta.detach(), eps=eps)
    dx, dgamma, dbeta = _maybe_call(instance_norm, "instance_norm_backward", dy, cache)

    _assert_allclose(y, y_ref.detach())
    _assert_allclose(dx, dx_ref)
    _assert_allclose(dgamma, dgamma_ref)
    _assert_allclose(dbeta, dbeta_ref)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "x_shape",
    [
        (4, 3, 10),  # (N, C, L) — 1D spatial
        (4, 3, 5, 6),  # (N, C, H, W) — 2D spatial
        (2, 3, 4, 5, 6),  # (N, C, D, H, W) — 3D spatial
    ],
    ids=["spatial_1d", "spatial_2d", "spatial_3d"],
)
def test_batch_norm_forward_backward_matches_torch(dtype, x_shape):
    """
    Skeleton for BatchNorm over channel dim (training mode, no running stats).

    PyTorch reference: torch.nn.functional.batch_norm in training mode

    Your implementation (expected signatures):
      - y, cache = batch_norm_forward(x, gamma, beta, eps=1e-5)
      - dx, dgamma, dbeta = batch_norm_backward(dy, cache)

    Notes:
    - Stats are over N and all spatial dimensions per channel (same reduction as `F.batch_norm`).
    """
    torch.manual_seed(0)

    C = x_shape[1]
    device = torch.device("cpu")
    x = torch.randn(x_shape, dtype=dtype, device=device, requires_grad=True)
    gamma = torch.randn(C, dtype=dtype, device=device, requires_grad=True)
    beta = torch.randn(C, dtype=dtype, device=device, requires_grad=True)
    dy = torch.randn(x_shape, dtype=dtype)
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

    y, cache = _maybe_call(batch_norm, "batch_norm_forward", x.detach(),
                                gamma.detach(), beta.detach(), eps=eps)
    dx, dgamma, dbeta = _maybe_call(batch_norm, "batch_norm_backward", dy, cache)

    _assert_allclose(y, y_ref.detach())
    _assert_allclose(dx, dx_ref)
    _assert_allclose(dgamma, dgamma_ref)
    _assert_allclose(dbeta, dbeta_ref)


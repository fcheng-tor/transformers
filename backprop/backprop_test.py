from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _get_module():
    # Pytest often prepends the test file's directory (e.g. `.../backprop`) to
    # sys.path. In that case, `backprop/backprop.py` is importable as the
    # top-level module `backprop` (not as `backprop.backprop`).
    return importlib.import_module("backprop")


def _maybe_call(mod, fn_name: str, *args, **kwargs):
    fn = getattr(mod, fn_name, None)
    if fn is None:
        pytest.skip(f"`{mod.__name__}.{fn_name}` not implemented yet. "
                    "Implement it and the test will start exercising it.")
    return fn(*args, **kwargs)


def _assert_allclose(a: torch.Tensor,
                     b: torch.Tensor,
                     *,
                     rtol=1e-4,
                     atol=1e-5):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_backprop_simple_affine_backward_matches_torch(dtype):
    """
    Skeleton test:
    - PyTorch reference: y = (x @ w) + b, loss = (y * dout).sum()
    - Your implementation: implement `affine_forward` and `affine_backward`

    Expected signatures (you can change, but then update this test):
      - y, cache = affine_forward(x, w, b)
      - dx, dw, db = affine_backward(dout, cache)
    """
    torch.manual_seed(0)

    N, D, M = 4, 8, 5
    device = torch.device("cpu")
    x = torch.randn(N, D, dtype=dtype, device=device, requires_grad=True)
    w = torch.randn(D, M, dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(M, dtype=dtype, device=device, requires_grad=True)
    dout = torch.randn(N, M, dtype=dtype)

    # PyTorch reference gradients
    y_ref = x @ w + b
    loss_ref = (y_ref * dout).sum()
    loss_ref.backward()
    dx_ref, dw_ref, db_ref = x.grad.detach(), w.grad.detach(), b.grad.detach()

    # Your implementation (skips until you implement)
    mod = _get_module()
    y, cache = _maybe_call(mod, "affine_forward", x.detach(), w.detach(),
                           b.detach())
    dx, dw, db = _maybe_call(mod, "affine_backward", dout, cache)

    _assert_allclose(y, y_ref.detach())
    _assert_allclose(dx, dx_ref)
    _assert_allclose(dw, dw_ref)
    _assert_allclose(db, db_ref)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_backprop_softmax_cross_entropy_backward_matches_torch(dtype):
    """
    Skeleton test for a common backprop building block.

    PyTorch reference:
      logits -> log_softmax -> nll_loss

    Your implementation:
      - loss, cache = softmax_cross_entropy_forward(logits, targets)
      - dlogits = softmax_cross_entropy_backward(cache)

    Notes:
    - Uses mean reduction like `torch.nn.functional.cross_entropy`.
    """
    torch.manual_seed(0)

    N, C = 6, 7
    device = torch.device("cpu")
    logits = torch.randn(N, C, dtype=dtype, device=device, requires_grad=True)
    targets = torch.randint(low=0, high=C, size=(N, ), dtype=torch.long)

    loss_ref = torch.nn.functional.cross_entropy(logits,
                                                 targets,
                                                 reduction="mean")
    loss_ref.backward()
    dlogits_ref = logits.grad.detach()

    mod = _get_module()
    loss, cache = _maybe_call(mod, "softmax_cross_entropy_forward",
                              logits.detach(), targets)
    dlogits = _maybe_call(mod, "softmax_cross_entropy_backward", cache)

    # loss can be python float, 0-d tensor, etc. Normalize to tensor.
    loss_t = torch.as_tensor(loss, dtype=dtype, device=device)
    _assert_allclose(loss_t.reshape(()),
                     loss_ref.detach().reshape(()),
                     rtol=1e-4,
                     atol=1e-5)
    _assert_allclose(dlogits, dlogits_ref)

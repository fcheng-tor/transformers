import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader import TRAIN_SPLIT, TinyShakespeareDataset


class SmallCNN(nn.Module):
    """Small image classifier with stacked convolutions."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class SmallTransformer(nn.Module):
    """Small sequence model with multi-head self-attention."""

    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 64,
        num_classes: int | None = None,
    ):
        super().__init__()
        num_classes = vocab_size if num_classes is None else num_classes
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        x = self.transformer(x)
        return self.head(x[:, -1, :])


class TrainingTransformer(SmallTransformer):
    """Transformer that predicts every token in the sequence (for LM training)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        x = self.transformer(x)
        return self.head(x)


class ShakespeareSequenceDataset(Dataset):
    """Contiguous character sequences built from TinyShakespeareDataset."""

    def __init__(self, seq_len: int):
        base = TinyShakespeareDataset()
        self.vocab_size = base.vocab_size()
        self.encoded = torch.tensor(base.encode(base.text), dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.encoded) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.encoded[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


@dataclass(frozen=True)
class Optimization:
    name: str
    description: str
    apply: Callable[[nn.Module, torch.Tensor | None], tuple[nn.Module, torch.Tensor | None]]


@dataclass(frozen=True)
class DataloaderConfig:
    name: str
    description: str
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
    non_blocking: bool = False


@dataclass
class TrainingTimings:
    total_ms: float
    dataloader_ms: float
    h2d_ms: float
    forward_ms: float
    backward_ms: float
    optimizer_ms: float


def _baseline(model: nn.Module, inputs: torch.Tensor | None) -> tuple[nn.Module, torch.Tensor | None]:
    return model, inputs


def _autocast_fp16(model: nn.Module, inputs: torch.Tensor | None) -> tuple[nn.Module, torch.Tensor | None]:
    return model, inputs


def _half_precision(model: nn.Module, inputs: torch.Tensor | None) -> tuple[nn.Module, torch.Tensor | None]:
    model = model.half()
    if inputs is not None and inputs.is_floating_point():
        inputs = inputs.half()
    return model, inputs


def _torch_compile(model: nn.Module, inputs: torch.Tensor | None) -> tuple[nn.Module, torch.Tensor | None]:
    return torch.compile(model), inputs


def _channels_last(model: nn.Module, inputs: torch.Tensor | None) -> tuple[nn.Module, torch.Tensor | None]:
    if inputs is not None and inputs.dim() == 4:
        return (
            model.to(memory_format=torch.channels_last),
            inputs.to(memory_format=torch.channels_last),
        )
    return model, inputs


def _cudnn_benchmark(model: nn.Module, inputs: torch.Tensor | None) -> tuple[nn.Module, torch.Tensor | None]:
    torch.backends.cudnn.benchmark = True
    return model, inputs


def _tf32_matmul(model: nn.Module, inputs: torch.Tensor | None) -> tuple[nn.Module, torch.Tensor | None]:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return model, inputs


def _inplace_relu(model: nn.Module, inputs: torch.Tensor | None) -> tuple[nn.Module, torch.Tensor | None]:
    if isinstance(model, nn.Sequential):
        return (
            nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 10),
            ),
            inputs,
        )
    return model, inputs


OPTIMIZATIONS: dict[str, Optimization] = {
    "baseline": Optimization(
        "baseline",
        "No extra optimizations; useful as a reference point.",
        _baseline,
    ),
    "autocast_fp16": Optimization(
        "autocast_fp16",
        "Run matmul/conv in fp16 via autocast while keeping sensitive ops in fp32.",
        _autocast_fp16,
    ),
    "half_precision": Optimization(
        "half_precision",
        "Cast model weights and inputs to float16 end-to-end.",
        _half_precision,
    ),
    "torch_compile": Optimization(
        "torch_compile",
        "Graph-compile the model with torch.compile for kernel fusion.",
        _torch_compile,
    ),
    "channels_last": Optimization(
        "channels_last",
        "Use NHWC memory layout; often helps conv-heavy models on NVIDIA GPUs.",
        _channels_last,
    ),
    "cudnn_benchmark": Optimization(
        "cudnn_benchmark",
        "Let cuDNN pick the fastest conv algorithm for fixed input shapes.",
        _cudnn_benchmark,
    ),
    "tf32": Optimization(
        "tf32",
        "Enable TF32 tensor cores for matmul/conv on Ampere+ GPUs.",
        _tf32_matmul,
    ),
    "inplace_relu": Optimization(
        "inplace_relu",
        "Use in-place activations to reduce memory traffic (MLP demo).",
        _inplace_relu,
    ),
}


DATALOADER_CONFIGS: dict[str, DataloaderConfig] = {
    "baseline": DataloaderConfig(
        "baseline",
        "Single-process loading on CPU; simplest but GPU may wait on data.",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        non_blocking=False,
    ),
    "workers": DataloaderConfig(
        "workers",
        "Prefetch batches in background worker processes.",
        num_workers=2,
        pin_memory=False,
        persistent_workers=False,
        non_blocking=False,
    ),
    "pin_memory": DataloaderConfig(
        "pin_memory",
        "Page-lock CPU tensors and use async host-to-device copies.",
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
        non_blocking=True,
    ),
    "full_pipeline": DataloaderConfig(
        "full_pipeline",
        "Workers + pin_memory + persistent_workers + non_blocking transfer.",
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        non_blocking=True,
    ),
}


def build_models() -> dict[str, tuple[Callable[[], nn.Module], torch.Tensor]]:
    return {
        "mlp": (
            lambda: nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 10)),
            torch.randn(128, 2048),
        ),
        "small_cnn": (
            lambda: SmallCNN(num_classes=10),
            torch.randn(64, 3, 32, 32),
        ),
        "small_transformer": (
            lambda: SmallTransformer(
                vocab_size=4096, d_model=128, nhead=4, num_layers=2, max_seq_len=64
            ),
            torch.randint(0, 4096, (32, 64)),
        ),
    }


def build_training_dataloader(
    dataloader_config: DataloaderConfig,
    batch_size: int,
    seq_len: int,
) -> tuple[DataLoader, int]:
    dataset = ShakespeareSequenceDataset(seq_len=seq_len)
    train_size = int(len(dataset) * TRAIN_SPLIT)
    eval_size = len(dataset) - train_size
    train_dataset, _ = random_split(
        dataset,
        lengths=[train_size, eval_size],
        generator=torch.Generator().manual_seed(42),
    )

    loader_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": dataloader_config.num_workers,
        "pin_memory": dataloader_config.pin_memory,
    }
    if dataloader_config.num_workers > 0:
        loader_kwargs["persistent_workers"] = dataloader_config.persistent_workers
        loader_kwargs["prefetch_factor"] = dataloader_config.prefetch_factor

    return DataLoader(train_dataset, **loader_kwargs), dataset.vocab_size


def reset_backends() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run_inference(
    model: nn.Module,
    inputs: torch.Tensor,
    optimization: Optimization,
) -> None:
    if optimization.name == "autocast_fp16":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = model(inputs)
        return

    _ = model(inputs)


def compute_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimization: Optimization,
) -> torch.Tensor:
    if optimization.name == "autocast_fp16":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(inputs)
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    logits = model(inputs)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_iter: Iterator,
    optimization: Optimization,
    dataloader_config: DataloaderConfig,
    profile: bool = False,
) -> tuple[torch.Tensor, TrainingTimings | None]:
    timings = TrainingTimings(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) if profile else None
    step_start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)

    if profile:
        load_start = time.perf_counter()
    with record_function("01_dataloading"):
        inputs, targets = next(data_iter)
    if profile and timings is not None:
        timings.dataloader_ms = (time.perf_counter() - load_start) * 1000

    if profile:
        h2d_start = time.perf_counter()
    with record_function("02_h2d_transfer"):
        inputs = inputs.cuda(non_blocking=dataloader_config.non_blocking)
        targets = targets.cuda(non_blocking=dataloader_config.non_blocking)
    if profile and timings is not None:
        timings.h2d_ms = (time.perf_counter() - h2d_start) * 1000

    if profile:
        forward_start = time.perf_counter()
    with record_function("03_forward"):
        loss = compute_loss(model, inputs, targets, optimization)
    if profile and timings is not None:
        timings.forward_ms = (time.perf_counter() - forward_start) * 1000

    if profile:
        backward_start = time.perf_counter()
    with record_function("04_backward"):
        loss.backward()
    if profile and timings is not None:
        timings.backward_ms = (time.perf_counter() - backward_start) * 1000

    if profile:
        optimizer_start = time.perf_counter()
    with record_function("05_optimizer"):
        optimizer.step()
    if profile and timings is not None:
        timings.optimizer_ms = (time.perf_counter() - optimizer_start) * 1000

    if profile and timings is not None:
        timings.total_ms = (time.perf_counter() - step_start) * 1000

    return loss, timings


def cycle_dataloader(dataloader: DataLoader) -> Iterator:
    while True:
        yield from dataloader


def warmup(
    model: nn.Module,
    inputs: torch.Tensor,
    optimization: Optimization,
    steps: int = 3,
) -> None:
    for _ in range(steps):
        run_inference(model, inputs, optimization)
    torch.cuda.synchronize()


def benchmark_ms(
    model: nn.Module,
    inputs: torch.Tensor,
    optimization: Optimization,
    steps: int = 20,
) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        run_inference(model, inputs, optimization)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms / steps


def benchmark_training_ms(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_iter: Iterator,
    optimization: Optimization,
    dataloader_config: DataloaderConfig,
    steps: int = 20,
) -> TrainingTimings:
    totals = TrainingTimings(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    torch.cuda.synchronize()
    for _ in range(steps):
        _, step_timings = training_step(
            model,
            optimizer,
            data_iter,
            optimization,
            dataloader_config,
            profile=True,
        )
        assert step_timings is not None
        totals.total_ms += step_timings.total_ms
        totals.dataloader_ms += step_timings.dataloader_ms
        totals.h2d_ms += step_timings.h2d_ms
        totals.forward_ms += step_timings.forward_ms
        totals.backward_ms += step_timings.backward_ms
        totals.optimizer_ms += step_timings.optimizer_ms
    torch.cuda.synchronize()

    scale = 1.0 / steps
    return TrainingTimings(
        total_ms=totals.total_ms * scale,
        dataloader_ms=totals.dataloader_ms * scale,
        h2d_ms=totals.h2d_ms * scale,
        forward_ms=totals.forward_ms * scale,
        backward_ms=totals.backward_ms * scale,
        optimizer_ms=totals.optimizer_ms * scale,
    )


def profile_model(
    model_name: str,
    model: nn.Module,
    inputs: torch.Tensor,
    optimization: Optimization,
    warmup_steps: int = 3,
    benchmark_steps: int = 20,
) -> float:
    reset_backends()
    model = model.cuda().eval()
    inputs = inputs.cuda()
    model, inputs = optimization.apply(model, inputs)

    warmup(model, inputs, optimization, steps=warmup_steps)
    avg_ms = benchmark_ms(model, inputs, optimization, steps=benchmark_steps)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function(f"{model_name}/{optimization.name}"):
            with torch.no_grad():
                run_inference(model, inputs, optimization)

    print(f"\n=== {model_name} | {optimization.name} ({avg_ms:.3f} ms/iter) ===")
    print(f"    {optimization.description}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return avg_ms


def profile_training_loop(
    optimization: Optimization,
    dataloader_config: DataloaderConfig,
    batch_size: int,
    seq_len: int,
    warmup_steps: int = 3,
    benchmark_steps: int = 10,
) -> TrainingTimings:
    reset_backends()

    dataloader, vocab_size = build_training_dataloader(
        dataloader_config,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    data_iter = cycle_dataloader(dataloader)

    model = TrainingTransformer(
        vocab_size=vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        max_seq_len=seq_len,
    ).cuda().train()
    model, _ = optimization.apply(model, None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for _ in range(warmup_steps):
        training_step(model, optimizer, data_iter, optimization, dataloader_config)
    torch.cuda.synchronize()

    timings = benchmark_training_ms(
        model,
        optimizer,
        data_iter,
        optimization,
        dataloader_config,
        steps=benchmark_steps,
    )

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function(f"training/{optimization.name}/{dataloader_config.name}"):
            training_step(model, optimizer, data_iter, optimization, dataloader_config)

    print(
        f"\n=== training | model={optimization.name} | dataloader={dataloader_config.name} "
        f"({timings.total_ms:.3f} ms/step) ==="
    )
    print(f"    Model opt: {optimization.description}")
    print(f"    Data opt:  {dataloader_config.description}")
    print(
        "    Phase breakdown:"
        f" dataloading={timings.dataloader_ms:.3f}ms"
        f" h2d={timings.h2d_ms:.3f}ms"
        f" forward={timings.forward_ms:.3f}ms"
        f" backward={timings.backward_ms:.3f}ms"
        f" optimizer={timings.optimizer_ms:.3f}ms"
    )
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))
    return timings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile inference and training pipelines for small PyTorch models.",
    )
    parser.add_argument(
        "--mode",
        choices=["inference", "training"],
        default="inference",
        help="inference: model-only profiling; training: full loop with TinyShakespeare data.",
    )
    parser.add_argument(
        "--model",
        choices=["mlp", "small_cnn", "small_transformer", "all"],
        default="all",
        help="Which model architecture to profile (inference mode only).",
    )
    parser.add_argument(
        "--optimization",
        choices=[*OPTIMIZATIONS.keys(), "all"],
        default="all",
        help="Which model optimization preset to apply.",
    )
    parser.add_argument(
        "--dataloader",
        choices=[*DATALOADER_CONFIGS.keys(), "all"],
        default="all",
        help="Which dataloader preset to use (training mode only).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (training mode only).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence length for Shakespeare training batches (training mode only).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help="Warmup iterations before profiling (use more for torch.compile).",
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=20,
        help="Timed iterations used for the ms/iter summary.",
    )
    return parser.parse_args()


def run_inference_mode(args: argparse.Namespace) -> None:
    models = build_models()
    model_names = list(models) if args.model == "all" else [args.model]
    optimization_names = list(OPTIMIZATIONS) if args.optimization == "all" else [args.optimization]

    results: list[tuple[str, str, float]] = []
    for model_name in model_names:
        make_model, inputs = models[model_name]
        for opt_name in optimization_names:
            optimization = OPTIMIZATIONS[opt_name]
            avg_ms = profile_model(
                model_name,
                make_model(),
                inputs.clone(),
                optimization,
                warmup_steps=args.warmup_steps,
                benchmark_steps=args.benchmark_steps,
            )
            results.append((model_name, opt_name, avg_ms))

    print("\n=== Summary (lower is better) ===")
    for model_name, opt_name, avg_ms in results:
        print(f"  {model_name:20s} {opt_name:16s} {avg_ms:8.3f} ms/iter")


def run_training_mode(args: argparse.Namespace) -> None:
    optimization_names = list(OPTIMIZATIONS) if args.optimization == "all" else [args.optimization]
    dataloader_names = list(DATALOADER_CONFIGS) if args.dataloader == "all" else [args.dataloader]

    results: list[tuple[str, str, TrainingTimings]] = []
    for opt_name in optimization_names:
        optimization = OPTIMIZATIONS[opt_name]
        for dl_name in dataloader_names:
            dataloader_config = DATALOADER_CONFIGS[dl_name]
            timings = profile_training_loop(
                optimization=optimization,
                dataloader_config=dataloader_config,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup_steps=args.warmup_steps,
                benchmark_steps=args.benchmark_steps,
            )
            results.append((opt_name, dl_name, timings))

    print("\n=== Training summary (lower is better) ===")
    for opt_name, dl_name, timings in results:
        print(
            f"  {opt_name:16s} {dl_name:16s} total={timings.total_ms:8.3f}ms"
            f"  load={timings.dataloader_ms:6.3f}  h2d={timings.h2d_ms:6.3f}"
            f"  fwd={timings.forward_ms:6.3f}  bwd={timings.backward_ms:6.3f}"
            f"  opt={timings.optimizer_ms:6.3f}"
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiler script.")

    args = parse_args()
    if args.mode == "inference":
        run_inference_mode(args)
    else:
        run_training_mode(args)


if __name__ == "__main__":
    main()

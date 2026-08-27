import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Imagenette
from torchvision.models import vit_b_16, ViT_B_16_Weights


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
DATA_ROOT = "./data"
OUTPUT_DIR = Path("results_exp3")
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
NUM_WORKERS = 0

PARTITIONS = [6, 9]
TOKEN_RATIOS = [0.75, 0.50]

PRECISIONS = [
    "FP32",
    "FP16",
    "INT8",
    "INT4"
]

LATENCY_WARMUP = 20
LATENCY_RUNS = 100

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.set_grad_enabled(False)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# SYSTEM INFO
# ============================================================

print("=" * 78)
print("Experiment 3: Intermediate Feature Quantization")
print("=" * 78)

print(f"PyTorch version : {torch.__version__}")
print(f"Device          : {device}")

if device.type == "cuda":
    print(
        f"GPU             : "
        f"{torch.cuda.get_device_name(0)}"
    )

print("=" * 78)


# ============================================================
# MODEL
# ============================================================

weights = ViT_B_16_Weights.DEFAULT

model = vit_b_16(
    weights=weights
).to(device)

model.eval()

preprocess = weights.transforms()

print("Model           : ViT-B/16")
print("Weights         : ImageNet-1K pretrained")


# ============================================================
# DATASET
# ============================================================

dataset = Imagenette(
    root=DATA_ROOT,
    split="val",
    size="160px",
    download=True,
    transform=preprocess
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"Validation images: {len(dataset)}")


# ============================================================
# IMAGENETTE -> IMAGENET LABEL MAPPING
# ============================================================

imagenette_names = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
]

imagenet_categories = weights.meta[
    "categories"
]

mapping = []

for name in imagenette_names:

    mapping.append(
        imagenet_categories.index(name)
    )

mapping = torch.tensor(
    mapping,
    dtype=torch.long,
    device=device
)


# ============================================================
# TOKEN PREPARATION
# ============================================================

def prepare_tokens(images):

    x = model._process_input(images)

    batch_size = x.shape[0]

    cls_token = model.class_token.expand(
        batch_size,
        -1,
        -1
    )

    x = torch.cat(
        [cls_token, x],
        dim=1
    )

    x = (
        x
        + model.encoder.pos_embedding
    )

    x = model.encoder.dropout(x)

    return x


# ============================================================
# TOKEN PRUNING
# ============================================================

def prune_tokens(x, rho):

    if rho >= 1.0:
        return x

    cls_token = x[:, :1, :]
    patch_tokens = x[:, 1:, :]

    n_patch = patch_tokens.shape[1]

    keep_count = max(
        1,
        int(round(rho * n_patch))
    )

    # L2-norm token saliency
    scores = torch.norm(
        patch_tokens,
        p=2,
        dim=2
    )

    indices = torch.topk(
        scores,
        k=keep_count,
        dim=1,
        largest=True,
        sorted=False
    ).indices

    # Preserve original order
    indices, _ = torch.sort(
        indices,
        dim=1
    )

    gather_indices = (
        indices
        .unsqueeze(-1)
        .expand(
            -1,
            -1,
            patch_tokens.shape[2]
        )
    )

    retained = torch.gather(
        patch_tokens,
        dim=1,
        index=gather_indices
    )

    return torch.cat(
        [cls_token, retained],
        dim=1
    )


# ============================================================
# FEATURE QUANTIZATION
# ============================================================

def quantize_dequantize(x, precision):
    """
    Quantize the transmitted intermediate representation
    and reconstruct it before cloud-side inference.

    FP32: unchanged.
    FP16: cast to float16 and restore float32.
    INT8/INT4: symmetric per-tensor quantization followed
    by dequantization.
    """

    if precision == "FP32":
        return x

    if precision == "FP16":

        return (
            x.to(torch.float16)
             .to(torch.float32)
        )

    if precision == "INT8":
        qmax = 127

    elif precision == "INT4":
        # signed 4-bit:
        # approximately [-7, 7]
        qmax = 7

    else:
        raise ValueError(
            f"Unsupported precision: {precision}"
        )

    max_abs = torch.amax(
        torch.abs(x),
        dim=(1, 2),
        keepdim=True
    )

    # avoid division by zero
    scale = max_abs / float(qmax)

    scale = torch.clamp(
        scale,
        min=1e-8
    )

    q = torch.round(
        x / scale
    )

    q = torch.clamp(
        q,
        -qmax,
        qmax
    )

    x_hat = q * scale

    return x_hat


# ============================================================
# SPLIT FORWARD PASS
# ============================================================

def forward_configuration(
    images,
    partition,
    rho,
    precision
):

    x = prepare_tokens(images)

    # Edge-side execution
    for idx in range(partition):
        x = model.encoder.layers[idx](x)

    # Proposed token pruning
    x = prune_tokens(
        x,
        rho
    )

    # Proposed transmitted-feature quantization
    x = quantize_dequantize(
        x,
        precision
    )

    # Cloud-side execution
    for idx in range(
        partition,
        len(model.encoder.layers)
    ):
        x = model.encoder.layers[idx](x)

    x = model.encoder.ln(x)

    cls = x[:, 0]

    logits = model.heads(cls)

    return logits


# ============================================================
# BASELINE ACCURACY
# ============================================================

def evaluate_baseline():

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(device)

            labels_1000 = mapping[labels]

            logits = model(images)

            pred = logits.argmax(dim=1)

            correct += (
                pred == labels_1000
            ).sum().item()

            total += labels.size(0)

    return (
        100.0 * correct / total
    )


# ============================================================
# CONFIGURATION ACCURACY
# ============================================================

def evaluate_configuration(
    partition,
    rho,
    precision
):

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(device)

            labels_1000 = mapping[labels]

            logits = forward_configuration(
                images,
                partition,
                rho,
                precision
            )

            pred = logits.argmax(dim=1)

            correct += (
                pred == labels_1000
            ).sum().item()

            total += labels.size(0)

    return (
        100.0 * correct / total
    )


# ============================================================
# LATENCY
# ============================================================

latency_input = torch.randn(
    1,
    3,
    224,
    224,
    device=device
)


def measure_latency(
    partition,
    rho,
    precision
):

    timings = []

    with torch.no_grad():

        for _ in range(
            LATENCY_WARMUP
        ):

            _ = forward_configuration(
                latency_input,
                partition,
                rho,
                precision
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        for _ in range(
            LATENCY_RUNS
        ):

            if device.type == "cuda":

                start_event = torch.cuda.Event(
                    enable_timing=True
                )

                end_event = torch.cuda.Event(
                    enable_timing=True
                )

                start_event.record()

                _ = forward_configuration(
                    latency_input,
                    partition,
                    rho,
                    precision
                )

                end_event.record()

                torch.cuda.synchronize()

                timings.append(
                    start_event.elapsed_time(
                        end_event
                    )
                )

            else:

                t0 = time.perf_counter()

                _ = forward_configuration(
                    latency_input,
                    partition,
                    rho,
                    precision
                )

                timings.append(
                    (
                        time.perf_counter()
                        - t0
                    )
                    * 1000.0
                )

    return {
        "mean_latency_ms":
            float(np.mean(timings)),

        "std_latency_ms":
            float(np.std(timings)),

        "median_latency_ms":
            float(np.median(timings)),

        "p95_latency_ms":
            float(
                np.percentile(
                    timings,
                    95
                )
            ),
    }


# ============================================================
# COMMUNICATION PAYLOAD
# ============================================================

TOTAL_PATCH_TOKENS = 196
TOKEN_DIM = 768

BIT_MAP = {
    "FP32": 32,
    "FP16": 16,
    "INT8": 8,
    "INT4": 4,
}

BASELINE_BYTES = (
    197
    * TOKEN_DIM
    * 32
    / 8
)


def communication_stats(
    rho,
    precision
):

    retained_patches = max(
        1,
        int(
            round(
                TOTAL_PATCH_TOKENS
                * rho
            )
        )
    )

    total_tokens = (
        retained_patches + 1
    )

    bits = BIT_MAP[
        precision
    ]

    total_bytes = (
        total_tokens
        * TOKEN_DIM
        * bits
        / 8
    )

    kb = (
        total_bytes / 1024
    )

    reduction = (
        1.0
        - total_bytes
        / BASELINE_BYTES
    ) * 100.0

    return (
        total_tokens,
        kb,
        reduction
    )


# ============================================================
# RUN
# ============================================================

baseline_accuracy = evaluate_baseline()

print(
    f"\nBaseline ViT-B/16 accuracy: "
    f"{baseline_accuracy:.3f}%"
)

results = []


for P in PARTITIONS:

    print("\n" + "=" * 78)
    print(f"Partition P = {P}")
    print("=" * 78)

    for rho in TOKEN_RATIOS:

        for precision in PRECISIONS:

            print(
                f"\nEvaluating "
                f"P={P}, "
                f"rho={rho:.2f}, "
                f"precision={precision}"
            )

            accuracy = (
                evaluate_configuration(
                    P,
                    rho,
                    precision
                )
            )

            latency = measure_latency(
                P,
                rho,
                precision
            )

            (
                total_tokens,
                comm_kb,
                comm_reduction
            ) = communication_stats(
                rho,
                precision
            )

            accuracy_drop = (
                baseline_accuracy
                - accuracy
            )

            results.append(
                {
                    "partition_P":
                        P,

                    "token_ratio_rho":
                        rho,

                    "precision":
                        precision,

                    "total_tokens":
                        total_tokens,

                    "accuracy_percent":
                        accuracy,

                    "accuracy_drop_pp":
                        accuracy_drop,

                    "mean_latency_ms":
                        latency[
                            "mean_latency_ms"
                        ],

                    "std_latency_ms":
                        latency[
                            "std_latency_ms"
                        ],

                    "median_latency_ms":
                        latency[
                            "median_latency_ms"
                        ],

                    "p95_latency_ms":
                        latency[
                            "p95_latency_ms"
                        ],

                    "communication_KB":
                        comm_kb,

                    "communication_reduction_percent":
                        comm_reduction,
                }
            )

            print(
                f"Accuracy         : "
                f"{accuracy:.3f}%"
            )

            print(
                f"Accuracy drop    : "
                f"{accuracy_drop:.3f} pp"
            )

            print(
                f"Mean latency     : "
                f"{latency['mean_latency_ms']:.3f} ms"
            )

            print(
                f"Tokens           : "
                f"{total_tokens}"
            )

            print(
                f"Communication    : "
                f"{comm_kb:.3f} KB"
            )

            print(
                f"Comm. reduction  : "
                f"{comm_reduction:.2f}%"
            )


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(results)

results_file = (
    OUTPUT_DIR
    / "exp3_quantization_results.csv"
)

df.to_csv(
    results_file,
    index=False
)


print("\n")
print("=" * 78)
print("EXPERIMENT 3 SUMMARY")
print("=" * 78)

summary_columns = [
    "partition_P",
    "token_ratio_rho",
    "precision",
    "total_tokens",
    "accuracy_percent",
    "accuracy_drop_pp",
    "mean_latency_ms",
    "communication_KB",
    "communication_reduction_percent",
]

print(
    df[
        summary_columns
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 78)
print(
    "Experiment 3 completed successfully."
)
print("=" * 78)

print(
    f"\nResults saved in:\n"
    f"{OUTPUT_DIR.resolve()}"
)

print(
    f"\nGenerated file:\n"
    f"{results_file.name}"
)

print("\nDone.")
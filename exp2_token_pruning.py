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
OUTPUT_DIR = Path("results_exp2")
OUTPUT_DIR.mkdir(exist_ok=True)

# Accuracy evaluation
BATCH_SIZE = 32
NUM_WORKERS = 0          # safest for Windows

# Candidate split points
PARTITIONS = [3, 6, 9]

# Token-retention ratios
TOKEN_RATIOS = [
    1.00,
    0.75,
    0.50,
    0.25
]

# Latency profiling
LATENCY_WARMUP = 30
LATENCY_RUNS = 150

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
# SYSTEM INFORMATION
# ============================================================

print("=" * 76)
print("Experiment 2: Token Pruning Accuracy and Efficiency")
print("=" * 76)

print(f"PyTorch version : {torch.__version__}")
print(f"Device          : {device}")

if device.type == "cuda":
    print(
        f"GPU             : "
        f"{torch.cuda.get_device_name(0)}"
    )

print("=" * 76)


# ============================================================
# LOAD PRETRAINED ViT-B/16
# ============================================================

weights = ViT_B_16_Weights.DEFAULT

model = vit_b_16(
    weights=weights
)

model = model.to(device)
model.eval()

preprocess = weights.transforms()

print("Model           : ViT-B/16")
print("Weights         : ImageNet-1K pretrained")


# ============================================================
# IMAGENETTE DATASET
# ============================================================

print("\nLoading Imagenette validation dataset...")

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
# IMAGENETTE LABEL -> IMAGENET-1K LABEL
# ============================================================
#
# Imagenette contains exactly these 10 ImageNet classes
# in this order.
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

imagenette_to_imagenet = []

for name in imagenette_names:

    if name not in imagenet_categories:
        raise RuntimeError(
            f"Could not find ImageNet category: {name}"
        )

    imagenette_to_imagenet.append(
        imagenet_categories.index(name)
    )

imagenette_to_imagenet = torch.tensor(
    imagenette_to_imagenet,
    dtype=torch.long,
    device=device
)

print("\nImagenette -> ImageNet-1K mapping")

for i, name in enumerate(
    imagenette_names
):
    print(
        f"{i:2d}  {name:20s}"
        f" -> ImageNet index "
        f"{imagenette_to_imagenet[i].item()}"
    )


# ============================================================
# TOKEN PREPARATION
# ============================================================

def prepare_tokens(images):
    """
    Reproduce torchvision ViT input processing
    up to the encoder blocks.
    """

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

    # positional embedding
    x = (
        x
        + model.encoder.pos_embedding
    )

    x = model.encoder.dropout(x)

    return x


# ============================================================
# TOKEN PRUNING
# ============================================================

def prune_tokens(x, retention_ratio):
    """
    Norm-based saliency pruning.

    CLS token is always retained.

    Patch-token importance:
        score_j = ||z_j||_2

    retention_ratio applies only to the 196
    patch tokens.
    """

    if retention_ratio >= 1.0:
        return x

    cls_token = x[:, :1, :]

    patch_tokens = x[:, 1:, :]

    num_patch_tokens = (
        patch_tokens.shape[1]
    )

    keep_count = max(
        1,
        int(
            round(
                retention_ratio
                * num_patch_tokens
            )
        )
    )

    # L2 norm of each patch token
    scores = torch.norm(
        patch_tokens,
        p=2,
        dim=2
    )

    # select most informative tokens
    top_indices = torch.topk(
        scores,
        k=keep_count,
        dim=1,
        largest=True,
        sorted=False
    ).indices

    # Sort retained indices so original
    # spatial/token order is preserved.
    top_indices, _ = torch.sort(
        top_indices,
        dim=1
    )

    gather_indices = (
        top_indices
        .unsqueeze(-1)
        .expand(
            -1,
            -1,
            patch_tokens.shape[2]
        )
    )

    retained_tokens = torch.gather(
        patch_tokens,
        dim=1,
        index=gather_indices
    )

    x_pruned = torch.cat(
        [
            cls_token,
            retained_tokens
        ],
        dim=1
    )

    return x_pruned


# ============================================================
# SPLIT + PRUNING FORWARD PASS
# ============================================================

def forward_pruned(
    images,
    partition,
    retention_ratio
):
    """
    Run blocks 1..P,
    prune tokens,
    run remaining blocks,
    then classification head.
    """

    x = prepare_tokens(images)

    # Edge-side blocks
    for idx in range(partition):
        x = model.encoder.layers[idx](x)

    # Proposed token pruning at split point
    x = prune_tokens(
        x,
        retention_ratio
    )

    # Cloud-side blocks
    for idx in range(
        partition,
        len(model.encoder.layers)
    ):
        x = model.encoder.layers[idx](x)

    # Final ViT encoder normalization
    x = model.encoder.ln(x)

    # CLS representation
    x = x[:, 0]

    # Classification head
    logits = model.heads(x)

    return logits


# ============================================================
# VALIDATE BASELINE FULL MODEL
# ============================================================

def evaluate_baseline():

    correct = 0
    total = 0

    print(
        "\nEvaluating unmodified "
        "ViT-B/16 baseline..."
    )

    with torch.no_grad():

        for batch_id, (
            images,
            labels
        ) in enumerate(loader):

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device
            )

            true_imagenet_labels = (
                imagenette_to_imagenet[
                    labels
                ]
            )

            logits = model(images)

            predictions = (
                logits.argmax(dim=1)
            )

            correct += (
                predictions
                == true_imagenet_labels
            ).sum().item()

            total += labels.size(0)

            if (
                (batch_id + 1) % 25 == 0
            ):
                print(
                    f"  processed "
                    f"{total}/{len(dataset)}"
                )

    accuracy = (
        100.0
        * correct
        / total
    )

    return accuracy


# ============================================================
# EVALUATE PRUNED CONFIGURATION
# ============================================================

def evaluate_configuration(
    partition,
    retention_ratio
):

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device
            )

            true_imagenet_labels = (
                imagenette_to_imagenet[
                    labels
                ]
            )

            logits = forward_pruned(
                images,
                partition,
                retention_ratio
            )

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions
                == true_imagenet_labels
            ).sum().item()

            total += labels.size(0)

    accuracy = (
        100.0
        * correct
        / total
    )

    return accuracy


# ============================================================
# LATENCY PROFILING
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
    retention_ratio
):

    with torch.no_grad():

        # warm-up
        for _ in range(
            LATENCY_WARMUP
        ):

            _ = forward_pruned(
                latency_input,
                partition,
                retention_ratio
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        timings = []

        for _ in range(
            LATENCY_RUNS
        ):

            if device.type == "cuda":

                start = torch.cuda.Event(
                    enable_timing=True
                )

                end = torch.cuda.Event(
                    enable_timing=True
                )

                start.record()

                _ = forward_pruned(
                    latency_input,
                    partition,
                    retention_ratio
                )

                end.record()

                torch.cuda.synchronize()

                timings.append(
                    start.elapsed_time(
                        end
                    )
                )

            else:

                t0 = time.perf_counter()

                _ = forward_pruned(
                    latency_input,
                    partition,
                    retention_ratio
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
            float(
                np.mean(timings)
            ),

        "std_latency_ms":
            float(
                np.std(timings)
            ),

        "median_latency_ms":
            float(
                np.median(timings)
            ),

        "p95_latency_ms":
            float(
                np.percentile(
                    timings,
                    95
                )
            ),
    }


# ============================================================
# COMMUNICATION VOLUME
# ============================================================

TOTAL_PATCH_TOKENS = 196
TOKEN_DIMENSION = 768

# Experiment 2 communication is FP32 only.
BITS_PER_ELEMENT = 32


def communication_stats(
    retention_ratio
):

    retained_patches = max(
        1,
        int(
            round(
                TOTAL_PATCH_TOKENS
                * retention_ratio
            )
        )
    )

    # CLS token is retained too
    total_tokens = (
        retained_patches + 1
    )

    total_bytes = (
        total_tokens
        * TOKEN_DIMENSION
        * BITS_PER_ELEMENT
        / 8
    )

    communication_kb = (
        total_bytes / 1024
    )

    baseline_bytes = (
        197
        * TOKEN_DIMENSION
        * BITS_PER_ELEMENT
        / 8
    )

    reduction = (
        1.0
        - total_bytes
        / baseline_bytes
    ) * 100.0

    return (
        retained_patches,
        total_tokens,
        communication_kb,
        reduction
    )


# ============================================================
# RUN EXPERIMENT
# ============================================================

baseline_accuracy = (
    evaluate_baseline()
)

print(
    f"\nBaseline accuracy: "
    f"{baseline_accuracy:.3f}%"
)


results = []


for partition in PARTITIONS:

    print(
        "\n"
        + "=" * 76
    )

    print(
        f"Partition P = "
        f"{partition}"
    )

    print(
        "=" * 76
    )

    for ratio in TOKEN_RATIOS:

        print(
            f"\nEvaluating "
            f"P={partition}, "
            f"rho={ratio:.2f}"
        )

        accuracy = (
            evaluate_configuration(
                partition,
                ratio
            )
        )

        latency = measure_latency(
            partition,
            ratio
        )

        (
            retained_patches,
            total_tokens,
            communication_kb,
            communication_reduction
        ) = communication_stats(
            ratio
        )

        accuracy_drop = (
            baseline_accuracy
            - accuracy
        )

        row = {
            "partition_P":
                partition,

            "token_ratio_rho":
                ratio,

            "retained_patch_tokens":
                retained_patches,

            "total_tokens_with_cls":
                total_tokens,

            "top1_accuracy_percent":
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

            "communication_FP32_KB":
                communication_kb,

            "communication_reduction_percent":
                communication_reduction,
        }

        results.append(row)

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
            f"{communication_kb:.3f} KB"
        )

        print(
            f"Comm. reduction  : "
            f"{communication_reduction:.2f}%"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_file = (
    OUTPUT_DIR
    / "exp2_token_pruning_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


baseline_df = pd.DataFrame(
    [
        {
            "model":
                "ViT-B/16",

            "dataset":
                "Imagenette validation",

            "images":
                len(dataset),

            "baseline_accuracy_percent":
                baseline_accuracy,
        }
    ]
)

baseline_file = (
    OUTPUT_DIR
    / "exp2_baseline_accuracy.csv"
)

baseline_df.to_csv(
    baseline_file,
    index=False
)


# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n")
print("=" * 76)
print("EXPERIMENT 2 SUMMARY")
print("=" * 76)

display_columns = [
    "partition_P",
    "token_ratio_rho",
    "total_tokens_with_cls",
    "top1_accuracy_percent",
    "accuracy_drop_pp",
    "mean_latency_ms",
    "communication_FP32_KB",
    "communication_reduction_percent",
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False
    )
)

print("\n")
print("=" * 76)
print(
    "Experiment 2 completed successfully."
)
print("=" * 76)

print(
    f"\nResults saved in:\n"
    f"{OUTPUT_DIR.resolve()}"
)

print("\nGenerated files:")
print(
    f"1. {results_file.name}"
)
print(
    f"2. {baseline_file.name}"
)

print("\nDone.")
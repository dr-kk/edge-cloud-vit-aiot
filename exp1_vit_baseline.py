import time
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
BATCH_SIZE = 1

FULL_WARMUP = 30
FULL_RUNS = 200

BLOCK_WARMUP = 20
BLOCK_RUNS = 100

SPLIT_POINTS = [3, 6, 9, 12]

OUT = Path("results_exp1")
OUT.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.set_grad_enabled(False)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# SYSTEM INFORMATION
# ============================================================

print("=" * 72)
print("Experiment 1: Baseline ViT-B/16 Profiling")
print("=" * 72)

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"Device          : {device}")
print(f"CPU             : {platform.processor()}")

if device.type == "cuda":
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    gpu_mem = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )
    print(f"GPU memory      : {gpu_mem:.2f} GB")

print("=" * 72)


# ============================================================
# MODEL
# ============================================================

weights = ViT_B_16_Weights.DEFAULT

model = vit_b_16(weights=weights)
model = model.to(device)
model.eval()

num_params = sum(
    p.numel() for p in model.parameters()
)

print("Model           : ViT-B/16")
print(f"Parameters      : {num_params / 1e6:.2f} M")


# ============================================================
# INPUT
# ============================================================

input_tensor = torch.randn(
    BATCH_SIZE,
    3,
    224,
    224,
    device=device
)


# ============================================================
# FULL MODEL PROFILING
# ============================================================

full_latencies = []

with torch.no_grad():

    for _ in range(FULL_WARMUP):
        _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(FULL_RUNS):

        if device.type == "cuda":

            start_event = torch.cuda.Event(
                enable_timing=True
            )
            end_event = torch.cuda.Event(
                enable_timing=True
            )

            start_event.record()
            _ = model(input_tensor)
            end_event.record()

            torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(
                end_event
            )

        else:

            start_time = time.perf_counter()

            _ = model(input_tensor)

            elapsed_ms = (
                time.perf_counter()
                - start_time
            ) * 1000.0

        full_latencies.append(
            elapsed_ms
        )


mean_latency = float(
    np.mean(full_latencies)
)

std_latency = float(
    np.std(full_latencies)
)

median_latency = float(
    np.median(full_latencies)
)

p95_latency = float(
    np.percentile(
        full_latencies,
        95
    )
)

fps = 1000.0 / mean_latency

peak_memory_mb = 0.0

if device.type == "cuda":
    peak_memory_mb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )


print("\nFULL MODEL RESULTS")
print("-" * 72)

print(f"Mean latency     : {mean_latency:.3f} ms")
print(f"Std latency      : {std_latency:.3f} ms")
print(f"Median latency   : {median_latency:.3f} ms")
print(f"P95 latency      : {p95_latency:.3f} ms")
print(f"Throughput       : {fps:.2f} FPS")
print(f"Peak GPU memory  : {peak_memory_mb:.2f} MB")


# ============================================================
# TOKEN PREPARATION
# ============================================================

def prepare_tokens(inp):

    z = model._process_input(inp)

    batch_size = z.shape[0]

    class_token = model.class_token.expand(
        batch_size,
        -1,
        -1
    )

    z = torch.cat(
        [class_token, z],
        dim=1
    )

    z = (
        z
        + model.encoder.pos_embedding
    )

    z = model.encoder.dropout(z)

    return z.detach().clone()


encoder_layers = model.encoder.layers

z = prepare_tokens(input_tensor)

print("\nTOKEN INPUT")
print("-" * 72)
print(f"Shape           : {list(z.shape)}")


# ============================================================
# BLOCK-LEVEL LATENCY PROFILING
# ============================================================

block_rows = []

with torch.no_grad():

    for block_id, block in enumerate(
        encoder_layers,
        start=1
    ):

        for _ in range(BLOCK_WARMUP):
            _ = block(z)

        if device.type == "cuda":
            torch.cuda.synchronize()

        block_times = []

        for _ in range(BLOCK_RUNS):

            if device.type == "cuda":

                start_event = torch.cuda.Event(
                    enable_timing=True
                )
                end_event = torch.cuda.Event(
                    enable_timing=True
                )

                start_event.record()

                output = block(z)

                end_event.record()

                torch.cuda.synchronize()

                elapsed_ms = (
                    start_event.elapsed_time(
                        end_event
                    )
                )

            else:

                start_time = time.perf_counter()

                output = block(z)

                elapsed_ms = (
                    time.perf_counter()
                    - start_time
                ) * 1000.0

            block_times.append(
                elapsed_ms
            )

        block_rows.append(
            {
                "block":
                    block_id,

                "mean_latency_ms":
                    float(
                        np.mean(
                            block_times
                        )
                    ),

                "std_latency_ms":
                    float(
                        np.std(
                            block_times
                        )
                    ),

                "median_latency_ms":
                    float(
                        np.median(
                            block_times
                        )
                    ),

                "p95_latency_ms":
                    float(
                        np.percentile(
                            block_times,
                            95
                        )
                    ),
            }
        )

        z = block(z).detach().clone()


block_df = pd.DataFrame(
    block_rows
)

block_file = (
    OUT / "vit_block_latency.csv"
)

block_df.to_csv(
    block_file,
    index=False
)

print("\nBLOCK LATENCIES")
print("-" * 72)

print(
    block_df.to_string(
        index=False
    )
)


# ============================================================
# PARTITION LATENCIES
# ============================================================

total_encoder_latency = float(
    block_df[
        "mean_latency_ms"
    ].sum()
)

cumulative_latency = 0.0

partition_rows = []

for _, row in block_df.iterrows():

    cumulative_latency += float(
        row["mean_latency_ms"]
    )

    partition_point = int(
        row["block"]
    )

    if partition_point in SPLIT_POINTS:

        cloud_latency = (
            total_encoder_latency
            - cumulative_latency
        )

        if partition_point == 12:
            cloud_latency = 0.0

        partition_rows.append(
            {
                "partition_P":
                    partition_point,

                "edge_encoder_latency_ms":
                    cumulative_latency,

                "cloud_encoder_latency_ms":
                    max(
                        cloud_latency,
                        0.0
                    ),

                "total_encoder_latency_ms":
                    total_encoder_latency,
            }
        )


partition_df = pd.DataFrame(
    partition_rows
)

partition_file = (
    OUT / "vit_partition_latency.csv"
)

partition_df.to_csv(
    partition_file,
    index=False
)

print("\nPARTITION LATENCIES")
print("-" * 72)

print(
    partition_df.to_string(
        index=False
    )
)


# ============================================================
# INTERMEDIATE REPRESENTATION SIZE
# ============================================================

TOKENS = 197
TOKEN_DIM = 768

precision_map = {
    "FP32": 32,
    "FP16": 16,
    "INT8": 8,
    "INT4": 4
}

representation_rows = []

for partition_point in SPLIT_POINTS:

    for precision, bits in precision_map.items():

        raw_bytes = (
            TOKENS
            * TOKEN_DIM
            * bits
            / 8
        )

        raw_kb = (
            raw_bytes / 1024
        )

        raw_mb = (
            raw_bytes
            / (1024 ** 2)
        )

        if partition_point == 12:
            transmitted_kb = 0.0
            transmitted_mb = 0.0
        else:
            transmitted_kb = raw_kb
            transmitted_mb = raw_mb

        representation_rows.append(
            {
                "partition_P":
                    partition_point,

                "tokens":
                    TOKENS,

                "token_dimension":
                    TOKEN_DIM,

                "precision":
                    precision,

                "bits_per_element":
                    bits,

                "raw_representation_KB":
                    raw_kb,

                "raw_representation_MB":
                    raw_mb,

                "transmitted_KB":
                    transmitted_kb,

                "transmitted_MB":
                    transmitted_mb,
            }
        )


representation_df = pd.DataFrame(
    representation_rows
)

representation_file = (
    OUT
    / "vit_intermediate_sizes.csv"
)

representation_df.to_csv(
    representation_file,
    index=False
)

print("\nINTERMEDIATE REPRESENTATION SIZES")
print("-" * 72)

print(
    representation_df.to_string(
        index=False
    )
)


# ============================================================
# BASELINE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    [
        {
            "model":
                "ViT-B/16",

            "parameters_M":
                num_params / 1e6,

            "batch_size":
                BATCH_SIZE,

            "input_resolution":
                "224x224",

            "full_warmup_runs":
                FULL_WARMUP,

            "full_timed_runs":
                FULL_RUNS,

            "mean_latency_ms":
                mean_latency,

            "std_latency_ms":
                std_latency,

            "median_latency_ms":
                median_latency,

            "p95_latency_ms":
                p95_latency,

            "fps":
                fps,

            "peak_gpu_memory_MB":
                peak_memory_mb,
        }
    ]
)

summary_file = (
    OUT / "vit_baseline_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# RAW FULL-MODEL LATENCY
# ============================================================

raw_latency_df = pd.DataFrame(
    {
        "run":
            np.arange(
                1,
                len(full_latencies) + 1
            ),

        "latency_ms":
            full_latencies,
    }
)

raw_latency_file = (
    OUT
    / "vit_raw_latency_runs.csv"
)

raw_latency_df.to_csv(
    raw_latency_file,
    index=False
)


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 72)
print("Experiment 1 completed successfully.")
print("=" * 72)

print(
    f"\nResults folder:\n{OUT.resolve()}"
)

print("\nGenerated files:")
print(f"1. {summary_file.name}")
print(f"2. {block_file.name}")
print(f"3. {partition_file.name}")
print(f"4. {representation_file.name}")
print(f"5. {raw_latency_file.name}")

print("\nDone.")
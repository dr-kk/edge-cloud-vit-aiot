from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

EXP3_FILE = Path(
    "results_exp3/exp3_quantization_results.csv"
)

OUTPUT_DIR = Path("results_exp4")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# NETWORK PROFILES
# ============================================================

BANDWIDTHS_MBPS = [
    5,
    10,
    25,
    50,
    100
]

NETWORK_DELAYS_MS = [
    10,
    25,
    50,
    100
]


# ============================================================
# REFERENCE BASELINE
# ============================================================
#
# Original uncompressed intermediate representation:
# 197 tokens x 768 dimensions x 32 bits
#
# Communication = 591 KB
#
# Baseline recognition accuracy from Exp-2/3:
# ============================================================

BASELINE_ACCURACY = 84.891720
BASELINE_COMM_KB = 591.0


# ============================================================
# LOAD EXPERIMENT 3 RESULTS
# ============================================================

if not EXP3_FILE.exists():
    raise FileNotFoundError(
        f"\nCannot find:\n{EXP3_FILE.resolve()}\n"
        "Run Experiment 3 first."
    )

df = pd.read_csv(EXP3_FILE)

print("=" * 80)
print("Experiment 4: Network-Aware Latency and Pareto Analysis")
print("=" * 80)

print(
    f"Loaded {len(df)} configurations "
    f"from Experiment 3."
)

print("\nInput columns:")
print(list(df.columns))


# ============================================================
# NETWORK LATENCY
# ============================================================

def transmission_latency_ms(
    communication_kb,
    bandwidth_mbps,
    network_delay_ms
):
    """
    Communication KB uses binary KB.

    Transfer time:
        bits / bits-per-second

    Convert to milliseconds.
    """

    bytes_to_send = (
        communication_kb * 1024.0
    )

    bits_to_send = (
        bytes_to_send * 8.0
    )

    bandwidth_bps = (
        bandwidth_mbps
        * 1_000_000.0
    )

    transfer_ms = (
        bits_to_send
        / bandwidth_bps
        * 1000.0
    )

    total_network_ms = (
        network_delay_ms
        + transfer_ms
    )

    return (
        transfer_ms,
        total_network_ms
    )


# ============================================================
# BUILD FULL NETWORK EXPERIMENT
# ============================================================

network_rows = []

for _, row in df.iterrows():

    P = int(
        row["partition_P"]
    )

    rho = float(
        row["token_ratio_rho"]
    )

    precision = row[
        "precision"
    ]

    accuracy = float(
        row["accuracy_percent"]
    )

    accuracy_drop = float(
        row["accuracy_drop_pp"]
    )

    compute_latency = float(
        row["mean_latency_ms"]
    )

    communication_kb = float(
        row["communication_KB"]
    )

    communication_reduction = float(
        row[
            "communication_reduction_percent"
        ]
    )

    for B in BANDWIDTHS_MBPS:

        for delay in NETWORK_DELAYS_MS:

            (
                transfer_ms,
                network_ms
            ) = transmission_latency_ms(
                communication_kb,
                B,
                delay
            )

            e2e_latency = (
                compute_latency
                + network_ms
            )

            network_rows.append(
                {
                    "partition_P":
                        P,

                    "token_ratio_rho":
                        rho,

                    "precision":
                        precision,

                    "accuracy_percent":
                        accuracy,

                    "accuracy_drop_pp":
                        accuracy_drop,

                    "compute_latency_ms":
                        compute_latency,

                    "communication_KB":
                        communication_kb,

                    "communication_reduction_percent":
                        communication_reduction,

                    "bandwidth_Mbps":
                        B,

                    "network_delay_ms":
                        delay,

                    "transfer_only_ms":
                        transfer_ms,

                    "total_network_ms":
                        network_ms,

                    "end_to_end_latency_ms":
                        e2e_latency,
                }
            )


network_df = pd.DataFrame(
    network_rows
)

network_file = (
    OUTPUT_DIR
    / "exp4_network_results.csv"
)

network_df.to_csv(
    network_file,
    index=False
)


# ============================================================
# PARETO FRONT FUNCTION
# ============================================================

def get_pareto_front(group):
    """
    Objectives:

    maximize accuracy
    minimize end-to-end latency
    minimize communication

    A point is Pareto optimal if no other point
    is:
      >= accuracy,
      <= latency,
      <= communication,
    with at least one strict improvement.
    """

    rows = group.reset_index(
        drop=True
    )

    pareto_flags = []

    for i, candidate in rows.iterrows():

        dominated = False

        for j, other in rows.iterrows():

            if i == j:
                continue

            accuracy_better_equal = (
                other["accuracy_percent"]
                >= candidate[
                    "accuracy_percent"
                ]
            )

            latency_better_equal = (
                other[
                    "end_to_end_latency_ms"
                ]
                <= candidate[
                    "end_to_end_latency_ms"
                ]
            )

            communication_better_equal = (
                other[
                    "communication_KB"
                ]
                <= candidate[
                    "communication_KB"
                ]
            )

            strictly_better = (
                other["accuracy_percent"]
                > candidate[
                    "accuracy_percent"
                ]
                or
                other[
                    "end_to_end_latency_ms"
                ]
                < candidate[
                    "end_to_end_latency_ms"
                ]
                or
                other[
                    "communication_KB"
                ]
                < candidate[
                    "communication_KB"
                ]
            )

            if (
                accuracy_better_equal
                and latency_better_equal
                and communication_better_equal
                and strictly_better
            ):
                dominated = True
                break

        pareto_flags.append(
            not dominated
        )

    result = rows.copy()

    result["pareto_optimal"] = (
        pareto_flags
    )

    return result


# ============================================================
# COMPUTE PARETO FRONT FOR EACH NETWORK PROFILE
# ============================================================

pareto_frames = []

for (
    bandwidth,
    delay
), group in network_df.groupby(
    [
        "bandwidth_Mbps",
        "network_delay_ms"
    ]
):

    pareto_group = get_pareto_front(
        group
    )

    pareto_frames.append(
        pareto_group
    )


pareto_df = pd.concat(
    pareto_frames,
    ignore_index=True
)

pareto_file = (
    OUTPUT_DIR
    / "exp4_pareto_results.csv"
)

pareto_df.to_csv(
    pareto_file,
    index=False
)


# ============================================================
# ACCURACY-CONSTRAINED BEST CONFIGURATION
# ============================================================
#
# High-accuracy operating condition:
# allow at most 1 percentage-point loss
# from baseline.
# ============================================================

MAX_ACCURACY_DROP_PP = 1.0

feasible = network_df[
    network_df[
        "accuracy_drop_pp"
    ] <= MAX_ACCURACY_DROP_PP
].copy()


best_rows = []

for (
    bandwidth,
    delay
), group in feasible.groupby(
    [
        "bandwidth_Mbps",
        "network_delay_ms"
    ]
):

    best_index = (
        group[
            "end_to_end_latency_ms"
        ].idxmin()
    )

    best = group.loc[
        best_index
    ]

    best_rows.append(
        best
    )


best_df = pd.DataFrame(
    best_rows
)

best_file = (
    OUTPUT_DIR
    / "exp4_best_accuracy_constrained.csv"
)

best_df.to_csv(
    best_file,
    index=False
)


# ============================================================
# COMMUNICATION-AGGRESSIVE OPERATING MODE
# ============================================================
#
# Allow maximum 1 percentage-point accuracy loss,
# then select minimum payload.
# Tie-break by end-to-end latency.
# ============================================================

aggressive_rows = []

for (
    bandwidth,
    delay
), group in feasible.groupby(
    [
        "bandwidth_Mbps",
        "network_delay_ms"
    ]
):

    sorted_group = group.sort_values(
        by=[
            "communication_KB",
            "end_to_end_latency_ms"
        ],
        ascending=[
            True,
            True
        ]
    )

    aggressive_rows.append(
        sorted_group.iloc[0]
    )


aggressive_df = pd.DataFrame(
    aggressive_rows
)

aggressive_file = (
    OUTPUT_DIR
    / "exp4_communication_aggressive.csv"
)

aggressive_df.to_csv(
    aggressive_file,
    index=False
)


# ============================================================
# PRINT SELECTED NETWORK PROFILE SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print(
    "BEST CONFIGURATION WITH <= 1 pp ACCURACY LOSS"
)
print("=" * 80)

columns_to_show = [
    "bandwidth_Mbps",
    "network_delay_ms",
    "partition_P",
    "token_ratio_rho",
    "precision",
    "accuracy_percent",
    "communication_KB",
    "end_to_end_latency_ms",
]

print(
    best_df[
        columns_to_show
    ].to_string(
        index=False
    )
)


# ============================================================
# PRINT PARETO COUNTS
# ============================================================

pareto_only = pareto_df[
    pareto_df[
        "pareto_optimal"
    ]
].copy()

pareto_count = (
    pareto_only
    .groupby(
        [
            "bandwidth_Mbps",
            "network_delay_ms"
        ]
    )
    .size()
    .reset_index(
        name="pareto_points"
    )
)

print("\n")
print("=" * 80)
print("PARETO FRONT SIZE BY NETWORK PROFILE")
print("=" * 80)

print(
    pareto_count.to_string(
        index=False
    )
)


pareto_only_file = (
    OUTPUT_DIR
    / "exp4_pareto_only.csv"
)

pareto_only.to_csv(
    pareto_only_file,
    index=False
)


# ============================================================
# SAVE NETWORK PROFILE TABLE
# ============================================================

profile_rows = []

for B in BANDWIDTHS_MBPS:

    for delay in NETWORK_DELAYS_MS:

        profile_rows.append(
            {
                "bandwidth_Mbps": B,
                "network_delay_ms": delay,
            }
        )


profile_df = pd.DataFrame(
    profile_rows
)

profile_file = (
    OUTPUT_DIR
    / "exp4_network_profiles.csv"
)

profile_df.to_csv(
    profile_file,
    index=False
)


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 80)
print("Experiment 4 completed successfully.")
print("=" * 80)

print(
    f"\nResults saved in:\n"
    f"{OUTPUT_DIR.resolve()}"
)

print("\nGenerated files:")

print(
    f"1. {network_file.name}"
)

print(
    f"2. {pareto_file.name}"
)

print(
    f"3. {pareto_only_file.name}"
)

print(
    f"4. {best_file.name}"
)

print(
    f"5. {aggressive_file.name}"
)

print(
    f"6. {profile_file.name}"
)

print("\nDone.")
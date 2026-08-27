from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

EXP2 = Path("results_exp2/exp2_token_pruning_results.csv")
EXP3 = Path("results_exp3/exp3_quantization_results.csv")
EXP4 = Path("results_exp4/exp4_network_results.csv")

OUT = Path("iotj_figures")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# ============================================================
# LOAD DATA
# ============================================================

df2 = pd.read_csv(EXP2)
df3 = pd.read_csv(EXP3)
df4 = pd.read_csv(EXP4)

print("CSV files loaded successfully.")

# ============================================================
# FIGURE 1
# TOKEN RETENTION vs ACCURACY
# ============================================================

fig, ax = plt.subplots(figsize=(3.5, 2.7))

markers = ["o", "s", "^"]

for marker, P in zip(markers, [3, 6, 9]):

    temp = df2[
        df2["partition_P"] == P
    ].sort_values("token_ratio_rho")

    ax.plot(
        temp["token_ratio_rho"],
        temp["top1_accuracy_percent"],
        marker=marker,
        linewidth=1.4,
        markersize=5,
        label=f"P={P}"
    )

ax.axhline(
    84.891720,
    linestyle="--",
    linewidth=1,
    label="Baseline"
)

ax.set_xlabel(r"Token Retention Ratio $\rho$")
ax.set_ylabel("Top-1 Accuracy (%)")

ax.set_xticks([
    0.25,
    0.50,
    0.75,
    1.00
])

ax.grid(
    True,
    linestyle=":",
    linewidth=0.6,
    alpha=0.7
)

ax.legend(frameon=False)

fig.tight_layout()

fig.savefig(
    OUT / "fig_accuracy_partition.pdf",
    bbox_inches="tight"
)

fig.savefig(
    OUT / "fig_accuracy_partition.png",
    bbox_inches="tight"
)

plt.close(fig)

print("Figure 1 generated.")


# ============================================================
# FIGURE 2
# ACCURACY vs COMMUNICATION PAYLOAD
# ============================================================

fig, ax = plt.subplots(figsize=(3.5, 2.7))

# P = 9 is the strongest partition
p9 = df3[
    df3["partition_P"] == 9
].copy()

precision_markers = {
    "FP32": "o",
    "FP16": "s",
    "INT8": "^",
    "INT4": "x"
}

for precision in [
    "FP32",
    "FP16",
    "INT8",
    "INT4"
]:

    temp = p9[
        p9["precision"] == precision
    ].sort_values(
        "communication_KB"
    )

    ax.scatter(
        temp["communication_KB"],
        temp["accuracy_percent"],
        marker=precision_markers[precision],
        s=45,
        label=precision
    )

# Original baseline
ax.scatter(
    [591.0],
    [84.891720],
    marker="D",
    s=48,
    label="Baseline"
)

# Highlight selected operating point
selected = p9[
    (p9["token_ratio_rho"] == 0.50)
    &
    (p9["precision"] == "INT8")
]

if len(selected) == 1:

    x = selected.iloc[0][
        "communication_KB"
    ]

    y = selected.iloc[0][
        "accuracy_percent"
    ]

    ax.annotate(
        "Selected\n(74.25 KB, 84.28%)",
        xy=(x, y),
        xytext=(125, 65),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=0.8
        ),
        fontsize=7
    )

ax.set_xlabel("Transmitted Payload (KB)")
ax.set_ylabel("Top-1 Accuracy (%)")

ax.grid(
    True,
    linestyle=":",
    linewidth=0.6,
    alpha=0.7
)

ax.legend(
    frameon=False,
    loc="lower right"
)

fig.tight_layout()

fig.savefig(
    OUT / "fig_accuracy_payload.pdf",
    bbox_inches="tight"
)

fig.savefig(
    OUT / "fig_accuracy_payload.png",
    bbox_inches="tight"
)

plt.close(fig)

print("Figure 2 generated.")


# ============================================================
# FIGURE 3
# NETWORK-AWARE END-TO-END LATENCY
# ============================================================

fig, ax = plt.subplots(figsize=(3.5, 2.7))

# Selected configuration:
# P=9, rho=0.50, INT8
selected_net = df4[
    (df4["partition_P"] == 9)
    &
    (df4["token_ratio_rho"] == 0.50)
    &
    (df4["precision"] == "INT8")
    &
    (df4["network_delay_ms"] == 10)
].copy()

selected_net = selected_net.sort_values(
    "bandwidth_Mbps"
)

ax.plot(
    selected_net["bandwidth_Mbps"],
    selected_net["end_to_end_latency_ms"],
    marker="o",
    linewidth=1.5,
    markersize=5,
    label=r"$P=9,\rho=0.50$, INT8"
)

# ------------------------------------------------------------
# Reference uncompressed representation
#
# We use the selected configuration's measured compute latency
# and replace only the payload with the original 591 KB.
# This isolates the communication-volume effect.
# ------------------------------------------------------------

baseline_payload_kb = 591.0
compute_latency_ms = 13.915923
network_delay_ms = 10.0

bandwidths = [
    5,
    10,
    25,
    50,
    100
]

baseline_e2e = []

for B in bandwidths:

    bits = (
        baseline_payload_kb
        * 1024
        * 8
    )

    transfer_ms = (
        bits
        /
        (B * 1_000_000)
        * 1000
    )

    total_ms = (
        compute_latency_ms
        + network_delay_ms
        + transfer_ms
    )

    baseline_e2e.append(
        total_ms
    )

ax.plot(
    bandwidths,
    baseline_e2e,
    marker="s",
    linestyle="--",
    linewidth=1.4,
    markersize=5,
    label="591-KB reference"
)

ax.set_xlabel("Available Bandwidth (Mbps)")
ax.set_ylabel("End-to-End Latency (ms)")

ax.set_xticks(
    bandwidths
)

ax.grid(
    True,
    linestyle=":",
    linewidth=0.6,
    alpha=0.7
)

ax.legend(
    frameon=False
)

fig.tight_layout()

fig.savefig(
    OUT / "fig_network_latency.pdf",
    bbox_inches="tight"
)

fig.savefig(
    OUT / "fig_network_latency.png",
    bbox_inches="tight"
)

plt.close(fig)


print("Figure 3 generated.")


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("IoT-J figures generated successfully")
print("=" * 70)

print(f"\nFolder: {OUT.resolve()}")

print("\nFiles:")
print("1. fig_accuracy_partition.pdf")
print("2. fig_accuracy_partition.png")
print("3. fig_accuracy_payload.pdf")
print("4. fig_accuracy_payload.png")
print("5. fig_network_latency.pdf")
print("6. fig_network_latency.png")

print("\nUse the PDF versions in the LaTeX manuscript.")
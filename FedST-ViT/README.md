\## Experimental Results and Discussion



FedST-ViT was evaluated for federated video action recognition on \*\*UCF101 Split 1\*\*, using a Video Swin Transformer backbone. The completed experiments compare FedST-ViT with FedAvg across IID and Dirichlet-based non-IID client partitions. The reported communication figures represent \*\*estimated uplink parameter payload\*\*, not measured end-to-end network traffic.



\### Overall Accuracy and Communication



| Data partition              | Method    | Final Top-1 accuracy | Estimated five-round uplink |

| --------------------------- | --------- | -------------------: | --------------------------: |

| IID, 5 clients              | FedAvg    |               93.55% |                 2663.43 MiB |

| IID, 5 clients              | FedST-ViT |               92.73% |                 1647.30 MiB |

| Non-IID, β = 0.1, 5 clients | FedAvg    |               85.41% |                 2663.43 MiB |

| Non-IID, β = 0.1, 5 clients | FedST-ViT |               82.10% |                 1566.72 MiB |

| Non-IID, β = 0.5, 5 clients | FedAvg    |               92.23% |                 2663.43 MiB |

| Non-IID, β = 0.5, 5 clients | FedST-ViT |               92.86% |                 1628.48 MiB |



FedST-ViT reduced estimated uplink payload by \*\*38.15% under IID\*\*, \*\*41.18% under β = 0.1\*\*, and \*\*38.86% under β = 0.5\*\*, relative to the normalized FedAvg parameter-only baseline.



The accuracy–communication trade-off depends on the client data distribution. Under IID partitioning, FedST-ViT achieved lower final-round Top-1 accuracy than FedAvg by approximately 0.82 percentage points. Under severe non-IID partitioning (β = 0.1), the accuracy difference increased to approximately 3.30 percentage points. Under β = 0.5, FedST-ViT achieved approximately 0.63 percentage points higher final-round accuracy while transmitting fewer parameter bytes. These observations are from the completed experimental runs and should not be interpreted as evidence of consistent accuracy improvement across all partitions.



\### Communication-Budget Sensitivity



The completed STU-only experiments under β = 0.5 and five clients show the following trade-off:



| Retention parameter γ | Final Top-1 accuracy | Estimated five-round uplink |

| --------------------: | -------------------: | --------------------------: |

|                  0.30 |               93.87% |                  917.61 MiB |

|                  0.40 |               93.81% |                 1056.66 MiB |

|                  0.50 |               93.55% |                 1308.63 MiB |

|                  0.60 |               93.52% |                 1593.55 MiB |



In these runs, smaller γ values corresponded to lower estimated uplink without a reduction in final-round accuracy. This is an \*\*observed result for the tested STU-only configuration\*\*, not a general claim that transmitting fewer parameters always improves accuracy.



\### Ablation Findings



For β = 0.5 and five clients, the STU-only configuration achieved \*\*93.52%\*\* final-round Top-1 accuracy, STU+ACB achieved \*\*92.44%\*\*, and the full STU+ACB+HAA configuration achieved \*\*92.86%\*\*. Thus, the full configuration did not achieve the highest accuracy in this single-seed ablation. Further repeated-seed experiments are needed to establish whether the observed differences are stable.



\### Qualitative Findings



A screening of 300 UCF101 evaluation clips produced 288 cases where both FedAvg and FedST-ViT predicted the correct class, three cases where only FedST-ViT was correct, two cases where only FedAvg was correct, and seven cases where both models were incorrect. The selected BalanceBeam, ApplyEyeMakeup, and ApplyLipstick examples illustrate both successful and unsuccessful FedST-ViT predictions. This screening is illustrative and is not a substitute for the full evaluation results.



\### Reproducibility and Limitations



The reported results come from completed experiments using a limited number of communication rounds and a single random seed per reported setting. The UCF101 Split-1 evaluation list was consulted during checkpoint selection and experimental development; consequently, these figures should \*\*not\*\* be described as performance on an untouched independent test set. Communication values are parameter-payload estimates that exclude downlink traffic, serialization overhead, and network latency. Independent test-set evaluation, repeated-seed analysis, additional datasets, and matched comparisons with other federated-learning methods remain future validation tasks.


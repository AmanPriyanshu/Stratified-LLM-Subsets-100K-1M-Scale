# Stratified LLM Subsets: Pre-Training, Instruction-Following, and Reasoning SFT Data at 100K-1M Scale

<div align="center">

[![Pre-Training](https://img.shields.io/badge/Pre--Training-FineWeb%20%26%20Proof--Pile-blue?style=for-the-badge)](https://huggingface.co/datasets/AmanPriyanshu/stratified-kmeans-diverse-pretraining-100K-1M)
[![Instruction-Following](https://img.shields.io/badge/Instruction--Following-Tulu%20%26%20Orca-green?style=for-the-badge)](https://huggingface.co/datasets/AmanPriyanshu/stratified-kmeans-diverse-instruction-following-100K-1M)
[![Reasoning](https://img.shields.io/badge/Reasoning-Llama--Nemotron-orange?style=for-the-badge)](https://huggingface.co/datasets/AmanPriyanshu/stratified-kmeans-diverse-reasoning-100K-1M)

[![Project Website](https://img.shields.io/badge/🌐_Project_Website-Visit-purple?style=for-the-badge)](https://amanpriyanshu.github.io/Stratified-LLM-Subsets-100K-1M-Scale/)

</div>

Stratified LLM Subsets delivers diverse training data at 100K-1M scales across pre-training (FineWeb-Edu, Proof-Pile-2), instruction-following (Tulu-3, Orca AgentInstruct), and reasoning distillation (Llama-Nemotron). Embedding-based k-means clustering ensures maximum diversity while re-balancing prevents category dominance across 5 high-quality open datasets.

---

## Overview

This project provides **diverse, representative subsets** from large-scale training corpora across multiple domains using **embedding-based k-means clustering** rather than random sampling:

- **Scales:** 50k, 100k, 250k, 500k, and 1M samples
- **Methodology:** Deterministic k-means clustering on embeddings (Snowflake Arctic-embed-xs) with 100 iterations
- **Balancing:** Square-root transformation for imbalanced datasets to prevent category dominance

---

## Datasets

### Pre-Training Dataset
**[stratified-kmeans-diverse-pretraining-100K-1M](https://huggingface.co/datasets/AmanPriyanshu/stratified-kmeans-diverse-pretraining-100K-1M)**

Combines FineWeb-Edu (educational web content) and Proof-Pile-2 (mathematical/scientific documents):
- **FineWeb-Edu:** 6 CommonCrawl snapshots from 2025 (99M rows filtered)
- **Proof-Pile-2:** algebraic-stack, arxiv, open-web-math

### Instruction-Following Dataset
**[stratified-kmeans-diverse-instruction-following-100K-1M](https://huggingface.co/datasets/AmanPriyanshu/stratified-kmeans-diverse-instruction-following-100K-1M)**

Combines Tulu-3 SFT Mixture and Orca AgentInstruct:
- **Tulu-3:** State-of-the-art post-training recipe (939K samples)
- **Orca AgentInstruct:** Agentic multi-step reasoning tasks (~1M samples)

### Reasoning Dataset
**[stratified-kmeans-diverse-reasoning-100K-1M](https://huggingface.co/datasets/AmanPriyanshu/stratified-kmeans-diverse-reasoning-100K-1M)**

Stratified subset of Llama-Nemotron Post-Training Dataset with square-root rebalancing:
- **Original:** 80.52% STEM dominated → **Rebalanced:** 51.81% STEM
- Categories: math, code, science, chat, safety

---

## Sampling Methodology

### Embedding-Based K-Means Clustering

1. **Embedding Generation:** Text embedded using Snowflake Arctic-embed-xs
2. **K-Means Clustering:** For M required samples, apply k-means with k=M clusters (100 iterations)
3. **Centroid Selection:** Select cluster centroids as representative samples
4. **Square-Root Balancing** (for imbalanced datasets):
   - Convert category counts to ratios
   - Apply sqrt transformation: `sqrt_ratio = sqrt(original_ratio)`
   - Renormalize: `balanced_ratio = sqrt_ratio / sum(sqrt_ratios)`

### Example: Llama-Nemotron Rebalancing

Original Llama-Nemotron Post-Training Dataset distribution was heavily skewed:
- **Math**: 66.96% → rebalanced to 52.03% (−22%)
- **Code**: 30.67% → rebalanced to 34.96% (+14%)
- **Science**: 2.15% → rebalanced to 9.26% (+330%)
- **Chat**: 0.12% → rebalanced to 2.15% (+1682%)
- **Safety**: 0.10% → rebalanced to 1.60% (+1580%)

Square-root transformation reduces math dominance while significantly increasing representation of underrepresented categories.

---

## Quick Start

```python
from datasets import load_dataset

# Load pre-training data
pretraining = load_dataset(
    "AmanPriyanshu/stratified-kmeans-diverse-pretraining-100K-1M", 
    split="100k"
)

# Load instruction-following data
instruction = load_dataset(
    "AmanPriyanshu/stratified-kmeans-diverse-instruction-following-100K-1M", 
    split="100k"
)

# Load reasoning data
reasoning = load_dataset(
    "AmanPriyanshu/stratified-kmeans-diverse-reasoning-100K-1M", 
    split="100k"
)
```

---

## Source Datasets

| Dataset | Task | License | Link |
|---------|------|---------|------|
| FineWeb-Edu | Pre-training | ODC-BY 1.0 | [HuggingFace](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| Proof-Pile-2 | Pre-training | Mixed | [HuggingFace](https://huggingface.co/datasets/EleutherAI/proof-pile-2) |
| Tulu-3 SFT | Instruction | ODC-BY 1.0 | [HuggingFace](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) |
| Orca AgentInstruct | Instruction | CDLA-Permissive 2.0 | [HuggingFace](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1) |
| Llama-Nemotron | Reasoning | CC BY 4.0 | [HuggingFace](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) |

---

## Citation

```bibtex
@misc{priyanshu2025stratifiedllm,
  title={{Stratified LLM Subsets: Pre-Training, Instruction-Following, and Reasoning SFT Data at 100K-1M Scale}},
  author={Priyanshu, Aman and Vijay, Supriti},
  year={2025},
  howpublished={\url{https://amanpriyanshu.github.io/Stratified-LLM-Subsets-100K-1M-Scale/}},
  note={Available at \url{https://huggingface.co/datasets/AmanPriyanshu/stratified-kmeans-diverse-reasoning-100K-1M}}
}
```

---

## License

Each subset inherits the license from its source datasets. Please refer to individual dataset cards for complete licensing terms.

---

## References

- **FineWeb-Edu**: [https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- **Proof-Pile-2**: [https://huggingface.co/datasets/EleutherAI/proof-pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2)
- **Tulu-3 SFT Mixture**: [https://huggingface.co/datasets/allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)
- **Orca AgentInstruct**: [https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1)
- **Llama-Nemotron Post-Training Dataset**: [https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset)

---

**Project Website:** [amanpriyanshu.github.io/Stratified-LLM-Subsets-100K-1M-Scale](https://amanpriyanshu.github.io/Stratified-LLM-Subsets-100K-1M-Scale/)

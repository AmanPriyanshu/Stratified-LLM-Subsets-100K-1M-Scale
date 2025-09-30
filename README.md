# Stratified LLM Subsets: Pre-Training, Instruction-Following, and Reasoning SFT data at 100K-1M Scale

Stratified LLM Subsets delivers balanced training data at 100K-1M scales across pre-training (FineWeb-Edu, Proof-Pile-2), instruction-following (Tulu-3, Orca AgentInstruct), and reasoning distillation (Nemotron). Square-root balancing prevents category dominance while maintaining diversity across 5 high-quality open datasets.

---

## Project Goal

This project aims to create **diverse, representative subsets** from large-scale training corpora across multiple domains. By intelligently sampling from six high-quality datasets spanning pre-training, instruction-following, and reasoning tasks, we generate balanced subsets at multiple scales:

- **25k, 50k, 125k, 250k, and 500k samples per dataset, except Nemotron which directly include sizes equivalent to total scales**
- **Total scales: 50k, 100k, 250k, 500k, and 1M samples across all tasks**

The sampling methodology aims to ensure diversity across categories while maintaining representativeness of the source distributions through square-root balancing techniques.

---

## Dataset Overview

| Name | Task | Link | License |
|------|------|------|---------|
| FineWeb-Edu | Pre-training | [HuggingFace](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | ODC-BY 1.0 |
| Proof-Pile-2 | Pre-training (Math/Science) | [HuggingFace](https://huggingface.co/datasets/EleutherAI/proof-pile-2) | Mixed (varies by subset) |
| Tulu-3 SFT Mixture | Instruction-following | [HuggingFace](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | ODC-BY 1.0 |
| Orca AgentInstruct 1M v1 | Instruction-following | [HuggingFace](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1) | CDLA-Permissive 2.0 |
| Nemotron Post-Training v1 | Post-training reasoning | [HuggingFace](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) | CC BY 4.0 |

---

## Dataset Descriptions

### 1. FineWeb-Edu (Pre-training)
**1.3 trillion tokens of high-quality educational web content**

FineWeb-Edu is a carefully filtered subset of CommonCrawl web data, processed through a custom educational quality classifier trained using Llama3-70B-Instruct annotations. The dataset filters out 92% of raw web data, retaining only content scoring 3+ on a 0-5 educational quality scale, resulting in exceptional performance on knowledge-intensive benchmarks compared to other open web datasets. Covers 96 CommonCrawl snapshots from 2013-2024 with extensive deduplication and quality filtering. Here, we limit to 99M rows specifically from those sourced in 2025: CC-MAIN-2025-05, CC-MAIN-2025-08, CC-MAIN-2025-13, CC-MAIN-2025-18, CC-MAIN-2025-21, and CC-MAIN-2025-26.

### 2. Proof-Pile-2 (Pre-training - Math/Science)
**55 billion tokens of mathematical and scientific documents**

A specialized pre-training corpus designed for mathematical reasoning, combining three major subsets: algebraic-stack (11B tokens) covering mathematical code including numerical computing, computer algebra, and formal mathematics; arxiv (29B tokens) with scientific papers; and open-web-math (15B tokens) featuring high-quality mathematical web text filtered from 200B+ HTML files. Covers formal mathematics (Lean, Coq, Isabelle, Agda), numerical computing (Julia, Python, MATLAB, R), and scientific literature.

### 3. Tulu-3 SFT Mixture (Instruction-following)
**939,344 prompts from state-of-the-art post-training recipe**

The Tulu-3 SFT mixture represents AllenAI's cutting-edge approach to instruction tuning, combining public datasets (57%) with synthetically generated skill-specific data (43%) using persona-driven methodology with GPT-4o and Claude 3.5 Sonnet. The mixture was carefully curated through iterative ablations to excel across diverse capabilities including precise instruction following, safety, and domain expertise. Trained models outperform Llama 3.1 Instruct, Qwen 2.5, and Mistral, with all data decontaminated against evaluation benchmarks.

### 4. Orca AgentInstruct 1M v1 (Instruction-following)
**~1 million high-quality instruction-following examples**

A large-scale instruction dataset created by Microsoft using an agentic framework to generate diverse, complex instructions. The dataset emphasizes multi-step reasoning, tool use, and realistic task decomposition, making it particularly valuable for training models that can follow complex, nuanced instructions. Commercially permissive license (CDLA-Permissive 2.0).

### 5. Nemotron Post-Training Dataset v1 (Post-training reasoning)
**~25.6 million samples across chat, code, math, STEM, and tool calling**

NVIDIA's comprehensive post-training dataset emphasizing verifiable reasoning and specialized capabilities across five distinct categories with a strong focus on STEM content. High-quality synthetic data generation with permissive CC BY 4.0 license.

---

## Sampling Methodology

### Category-Aware Balanced Sampling

For datasets with pre-existing category splits (**Nemotron** and **Proof-Pile-2**), we apply a **square-root balancing transformation** to prevent over-representation of dominant categories while maintaining category signal.

#### Nemotron Category Rebalancing (End-to-End Example)

**Step 1: Raw Sample Counts**
```
chat:         746,622
code:       1,896,395
math:       2,044,407
stem:      20,662,167
tool_calling: 310,051
────────────────────────
Total:     25,659,642
```

**Step 2: Convert to Ratios**
```python
ratios = {
    "chat": 746622 / 25659642 = 0.0291
    "code": 1896395 / 25659642 = 0.0739
    "math": 2044407 / 25659642 = 0.0797
    "stem": 20662167 / 25659642 = 0.8052
    "tool_calling": 310051 / 25659642 = 0.0121
}
```

**Step 3: Apply Square-Root Transform (α = 0.5)**
```python
import math
sqrt_values = {
    "chat": math.sqrt(0.0291) = 0.1705
    "code": math.sqrt(0.0739) = 0.2719
    "math": math.sqrt(0.0797) = 0.2823
    "stem": math.sqrt(0.8052) = 0.8973
    "tool_calling": math.sqrt(0.0121) = 0.1100
}
sqrt_sum = 0.1705 + 0.2719 + 0.2823 + 0.8973 + 0.1100 = 1.7320
```

**Step 4: Renormalize to Sum to 1.0**
```python
balanced_ratios = {
    "chat": 0.1705 / 1.7320 = 0.0985
    "code": 0.2719 / 1.7320 = 0.1570
    "math": 0.2823 / 1.7320 = 0.1630
    "stem": 0.8973 / 1.7320 = 0.5181
    "tool_calling": 0.1100 / 1.7320 = 0.0635
}
```

**Final Balanced Distribution:**
```
chat:         9.85%  (was 2.91% → +238% increase)
code:        15.70%  (was 7.39% → +112% increase)
math:        16.30%  (was 7.97% → +105% increase)
stem:        51.81%  (was 80.52% → -36% decrease)
tool_calling: 6.35%  (was 1.21% → +425% increase)
```

This transformation:
- Prevents STEM's 80% dominance from overwhelming the subset
- Elevates underrepresented categories (chat, tool_calling) significantly
- Maintains relative ordering while improving balance
- Preserves category diversity signals through principled reweighting

#### Proof-Pile-2 Category Balancing

The same square-root balancing methodology is applied to:

**Proof-Pile-2 subsets:**
- `algebraic-stack` (mathematical code)
- `arxiv` (scientific papers)
- `open-web-math` (mathematical web text)

### Unified Corpus Sampling

For datasets without pre-existing category splits (**FineWeb-Edu**, **Tulu-3**, **Orca AgentInstruct**), we treat them as unified corpora and sample uniformly across all examples while maintaining diversity through Random stratified sampling.

---

## Target Subset Scales

We generate multiple scales of balanced subsets to support various training scenarios:

| Per-Dataset Size | Total Task Size | Use Case |
|---------------|---------------------|----------|
| 25k | 50k | Rapid prototyping, ablations |
| 50k | 100k | Small-scale fine-tuning |
| 125k | 250k | Medium-scale experiments |
| 250k | 500k | Production small models |
| 500k | 1M | Production medium models |

Each subset maintains:
- Balanced category representation (where applicable)
- Diversity across all source datasets (however, we take task size values directly for Nemotron as that's the only reasoning dataset we incorporate.)

---

## Citation

If you use these subsets, please cite:

```bibtex
@misc{priyanshu2025stratifiedllm,
  title={{Stratified LLM Subsets: Pre-Training, Instruction-Following, and Reasoning SFT Data at 100K-1M Scale}},
  author={Priyanshu, Aman and Vijay, Supriti},
  year={2025},
  howpublished={\url{https://amanpriyanshu.github.io/Stratified-LLM-Subsets-100K-1M-Scale/}},
  note={Multi-scale balanced sampling across six diverse training corpora}
}
```

Please also cite the original source datasets according to their respective licenses and attribution requirements.

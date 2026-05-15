# Fine-Tuning Mistral-7B with QLoRA

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-FFD21E?style=flat)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Parameter-efficient supervised fine-tuning of Mistral-7B-v0.1 using QLoRA (Quantized Low-Rank Adaptation)**
*Instruction following · Dolly-15k · NF4 Quantization · PEFT*

</div>

---

## Abstract

This project implements **QLoRA** to perform parameter-efficient fine-tuning of **Mistral-7B-v0.1** for instruction-following tasks. By quantizing the base model to **4-bit precision** (NF4) and injecting trainable low-rank adapter weights only into attention projections, we achieve effective adaptation with a fraction of the memory footprint required by full fine-tuning.

> **TL;DR** — Fine-tune a 7B LLM with ~6GB VRAM via QLoRA. Perplexity **4.00**, ROUGE-1 **0.364**, ROUGE-L **0.291**.

---

## Method

### BitsAndBytes (NF4 Quantization)

| Config | Value |
|---|---|
| `load_in_4bit` | True |
| `bnb_4bit_quant_type` | nf4 |
| `bnb_4bit_use_double_quant` | True |

### LoRA Configuration

| Hyperparameter | Value |
|---|---|
| `r` (rank) | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| `target_modules` | q_proj, v_proj |
| `bias` | none |

### Training

| Hyperparameter | Value |
|---|---|
| `learning_rate` | 2e-4 |
| `max_steps` | 10 |
| `batch_size` | 4 |
| `gradient_accumulation_steps` | 2 |
| `fp16` | True |

---

## Results

| Metric | Value |
|---|---|
| **Perplexity** | 4.00 |
| **ROUGE-1 (F1)** | 0.364 |
| **ROUGE-L (F1)** | 0.291 |

---

## Dataset

[Databricks Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) — 15k human-generated instruction/response pairs.

---

## Installation

```bash
pip install torch transformers peft trl bitsandbytes datasets accelerate rouge_score
```

---

## References

```bibtex
@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}
```

---

**Gabriel Wamat** · [GitHub](https://github.com/Gabriel-Wamat)

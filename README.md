# Adaptive Block Routing between Attention and State-Space Models

## Overview

This repository implements a hybrid sequence modeling architecture that dynamically routes input segments between:

* Transformer (attention-based models for global context)
* State-Space Models (Mamba) for efficient sequential processing

We introduce a dynamic routing mechanism based on block-level decisions using Straight-Through Gumbel-Softmax, enabling discrete expert selection while maintaining differentiability and end-to-end trainability.

---

## Motivation

Transformers and State-Space Models (SSMs) exhibit complementary strengths:

| Model       | Strength                        | Limitation               |
| ----------- | ------------------------------- | ------------------------ |
| Transformer | Global dependency modeling      | High computational cost  |
| Mamba (SSM) | Efficient sequential processing | Limited global reasoning |

This work investigates whether dynamic routing can combine these strengths by selecting the appropriate computation pathway for different parts of a sequence.

---

## Key Contributions

* Block-level routing instead of token-level routing for improved efficiency and structural coherence
* Straight-Through Gumbel-Softmax for discrete yet differentiable routing decisions
* Hybrid architecture combining Transformer and State-Space Models
* Empirical analysis of performance–efficiency trade-offs
* Study of routing granularity and its impact on hardware utilization

---

## Architecture

### Pipeline

1. Input sequence is divided into blocks
2. Each block is summarized using mean pooling
3. A routing network predicts whether to use:

   * Transformer
   * Mamba
4. Selected model processes each block
5. Outputs are recombined into the original sequence

---

### Routing Mechanism

* Linear projection produces routing logits
* Gumbel-Softmax enables stochastic exploration
* Straight-through estimator provides hard decisions during forward pass
* Load balancing and entropy regularization stabilize training

---

## Experimental Results

### Latency Benchmark

| Model                           | Latency (seconds) |
| ------------------------------- | ----------------- |
| Transformer                     | 0.315             |
| Mamba                           | 0.172             |
| Soft Hybrid                     | 0.140             |
| Sparse Hybrid (block size = 2)  | 0.646             |
| Sparse Hybrid (block size = 8)  | 0.195             |
| Sparse Hybrid (block size = 16) | 0.195             |

---

### Language Modeling (Loss)

| Model         | Final Loss |
| ------------- | ---------- |
| Transformer   | 1.01       |
| Mamba         | 0.12       |
| Soft Hybrid   | 0.086      |
| Sparse Hybrid | 0.48       |

---

## Key Insights

* Sparse routing does not guarantee efficiency due to GPU memory and kernel overhead
* Dense computation patterns are often more hardware-efficient
* Routing granularity controls the trade-off between flexibility and efficiency
* Block-level routing provides a better balance than token-level routing
* Hybrid routing introduces a trade-off between performance and computational cost

---

## Visualizations

* Latency vs Block Size
* Loss vs Latency Trade-off

(Available in `experiments/exp1/`)

---

## Project Structure

```
models/
 ├── transformer.py
 ├── mamba.py
 ├── router.py
 ├── hybrid_model.py
 ├── transformer_only.py
 ├── mamba_only.py
 ├── soft_hybrid.py
 └── lm_wrapper.py

training/
 ├── train_lm.py
 └── language_model.py

experiments/
 └── exp1/
      ├── benchmark.py
      ├── plot.py
      └── plot_tradeoff.py

data/
 └── sample.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

### Mamba Installation (GPU Required)

```bash
pip install mamba-ssm
```

---

## Usage

### Run Benchmark

```bash
python -m experiments.exp1.benchmark
```

---

### Train Language Model

```bash
python -m training.train_lm
```

---

### Generate Plots

```bash
python experiments/exp1/plot.py
python experiments/exp1/plot_tradeoff.py
```

---

## Limitations

* Experiments are conducted on a small dataset for demonstration purposes
* Sparse execution introduces overhead due to dynamic memory operations
* Dense hybrid models achieve better performance but require higher computation

---

## Future Work

* Evaluation on large-scale datasets (e.g., WikiText, OpenWebText)
* Hardware-aware optimizations using Triton or CUDA kernels
* Multi-layer hybrid architectures
* Adaptive routing across layers

---

## Research Direction

This work highlights the importance of co-designing model architecture and execution strategies to align with modern hardware characteristics.

---

## Acknowledgements

* PyTorch
* Mamba State-Space Models
* Transformer architectures

---

## Contact

For questions or collaboration, please open an issue or reach out.

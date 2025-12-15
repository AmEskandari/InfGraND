# InfGraND: An Influence-Guided GNN-to-MLP Knowledge Distillation

<p align="center">
  <img src="assets/infgrand_overview.png" width="800"/>
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=lfzHR3YwlD"><img src="https://img.shields.io/badge/TMLR-OpenReview-b31b1b.svg" alt="TMLR"></a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

Official PyTorch implementation of **"InfGraND: An Influence-Guided GNN-to-MLP Knowledge Distillation"**, published in *Transactions on Machine Learning Research (TMLR)*, 2025.

---

## 🚧 Code Coming Soon

We are currently preparing the codebase for public release. The complete implementation will be available shortly.

**Stay tuned!** ⭐ Star this repository to get notified when the code is released.

---

## Abstract

Graph Neural Networks (GNNs) are the go-to model for graph data analysis. However, GNNs rely on two key operations—aggregation and update, which can pose challenges for low-latency inference tasks or resource-constrained scenarios. Simple Multi-Layer Perceptrons (MLPs) offer a computationally efficient alternative. Yet, training an MLP in a supervised setting often leads to suboptimal performance. Knowledge Distillation (KD) from a GNN teacher to an MLP student has emerged to bridge this gap. However, most KD methods either transfer knowledge uniformly across all nodes or rely on graph-agnostic indicators such as prediction uncertainty. We argue this overlooks a more fundamental, graph-centric inquiry: *"How important is a node to the structure of the graph?"* We introduce **InfGraND**, an Influence-guided Graph Knowledge Distillation from GNN to MLP that addresses this by identifying and prioritizing structurally influential nodes to guide the distillation process, ensuring that the MLP learns from the most critical parts of the graph.

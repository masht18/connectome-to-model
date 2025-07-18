# 🧠 Connectome-to-Model: Neural Architecture from Biological Connectivity

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Transform biological connectome data into functional artificial neural networks with biologically-inspired top-down feedback mechanisms**

[Installation](#-installation) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Examples](#-examples) • [API Reference](#-api-reference)

</div>

---

## 🌟 Overview


This codebase provides a framework for converting simple connectome data (structured connectivity between brain areas) into functional artificial neural networks with top-down feedback. The resulting models incorporate biologically-inspired feedback mechanisms and can process multiple input streams simultaneously.


👉 **See the interactive [MNIST Jupyter tutorial](mnist_graph_training.ipynb) for a hands-on example.**


## ✨ Key Features

- 🔄 **Recurrent ConvGRU cells** with biological top-down feedback
- 🎯 **Multiple feedback mechanisms**: Multiplicative, additive, and basal
- 📊 **CSV-based connectome specification** For ANNs with complex connectivity
- 🔌 **Multi-input support**: Process visual, auditory, and other modalities

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Conda or pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/masht18/convgru_feedback.git
   cd convgru_feedback
   ```

2. **Create conda environment**
   ```bash
   conda create -n connectome python=3.8 scipy matplotlib pandas
   conda activate connectome
   ```

3. **Install PyTorch**
   ```bash
   # For CUDA 11.x
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CPU only
   pip3 install torch torchvision torchaudio
   ```

4. **Install the package**
   ```bash
   pip install -e .
   ```

## 🎯 Quick Start

### Basic Usage

```python
from connectome_to_model.model.graph import Graph
from connectome_to_model.model.architectures import ConnectomicsConvGRU

# Define which nodes receive input and provide output
input_nodes = [0, 2]  # V1 and A1 (visual and auditory)
output_nodes = [1]    # Higher-level area

# Load connectome from CSV
graph = Graph('path/to/connectome.csv', 
              input_nodes=input_nodes, 
              output_nodes=output_nodes)

# Get input specifications from graph
input_sizes = graph.find_input_sizes()  
input_dims = graph.find_input_dims()

# Build the neural network
model = ConnectomicsConvGRU(graph, input_sizes, input_dims,
                           topdown=True,
                           dropout=True,
                           proj_hidden_dim=32).cuda()

# Optional: process multiple inputs
visual_input = torch.randn(batch_size, seq_len, 3, 32, 32).cuda()
audio_input = torch.randn(batch_size, seq_len, 1, 64, 64).cuda()
outputs = model([visual_input, audio_input])
```

## 📊 Connectome File Format

The connectome CSV file defines both connectivity and area properties:

### Example: Simple A→B→C Architecture

| A | B | C | hidden_dim | input_dim | input_h | input_w | kernel_h | kernel_w | basal_topdown_dim | apical_topdown_dim |
|---|---|---|------------|-----------|---------|---------|----------|----------|-------------------|---------------------|
| 0 | 1 | 0 | 10 | 10 | 32 | 32 | 3 | 3 | 0 | 1 |
| 0 | 0 | 1 | 10 | 10 | 32 | 32 | 3 | 3 | 0 | 1 |
| 0 | 0 | 0 | 10 | 10 | 32 | 32 | 3 | 3 | 0 | 1 |

### Column Descriptions

| Column | Description | Example Values |
|--------|-------------|----------------|
| **Connectivity** | First N columns define connections (1/0) | 0 or 1 |
| **hidden_dim** | Number of channels in the area | 10-512 |
| **input_dim** | Channels for bottom-up input | 1-256 |
| **input_h/w** | Spatial dimensions of input | 28, 32, 64 |
| **kernel_h/w** | Receptive field size | 3, 5, 7 |
| **basal_topdown_dim** | Basal feedback channels | 0-64 |
| **apical_topdown_dim** | Apical feedback channels | 0-64 |

## 🔄 Top-Down Feedback Mechanisms

### 1. Multiplicative/Gain-Modulating Feedback (Default)

- **Function**: Amplifies existing signals
- **Enable**: Default behavior (set `topdown=True`)


### 2. Threshold-Shifting/Composite Feedback

- **Function**: In addition to default modulatory feedback, this type of feedback can also activate silent neurons. Its strength compared to feedforward input depends on apical_topdown_dim compared to hidden_dim
- **Enable**: Set `apical_topdown_dim > 0` in connectome csv


### 3. Basal Feedback

- **Function**: Earlier additive integration of bottom-up signals. Experimental.
- **Biological Basis**: Experimental functionality based on feedforward-like projections from high order to low order mice brain regions in [Harris et al., Nature 2019](https://www.nature.com/articles/s41586-019-1716-z)
- **Enable**: Set `basal_topdown_dim > 0` in connectome csv

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
 @article{Tugsbayar_2025, 
 title={Top-down feedback matters: Functional impact of brainlike connectivity motifs on audiovisual integration}, 
 url={http://dx.doi.org/10.7554/eLife.105953.1}, 
 DOI={10.7554/elife.105953.1}, 
 publisher={eLife Sciences Publications, Ltd}, 
 author={Tugsbayar, Mashbayar and Li, Mingze and Muller, Eilif B and Richards, Blake}, year={2025}, month=apr }

```
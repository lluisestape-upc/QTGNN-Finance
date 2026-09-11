# QTGNN-Finance

[![Report](https://img.shields.io/badge/Report-PDF-red)](report.pdf)

A hybrid Quantum-Temporal Graph Neural Network pipeline for stock price prediction, combining quantum computing, graph learning, and NLP sentiment analysis.

## Setup

Install PyTorch first, matched to the machine's CUDA version, then the rest:

```bash
# pick one
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
python main.py
```

`torch-geometric` resolves against the installed torch, so the order matters.
Price data is pulled from `yfinance` and FinBERT from the HuggingFace hub at
runtime, so there is nothing else to download by hand.

### Devices

| Variable | Default | Notes |
|---|---|---|
| `QTGNN_DEVICE` | `cuda` if available, else `cpu` | classical layers (GRU, attention, readout) |
| `QTGNN_QDEVICE` | `default.qubit` | PennyLane simulator backing the message circuit |

The circuit is evaluated with **parameter broadcasting over all edges at once**,
which is what makes `default.qubit` the fastest choice here — `lightning.qubit`
does not broadcast and measures ~20x slower in that regime, despite being faster
on a per-edge loop. Benchmark before switching.

Note that the classical half of the model is tiny; the simulated circuit is the
bottleneck, so a larger GPU helps far less than you would expect.

## Features

### Quantum Graph Attention Convolution (QGATConv)
A custom `MessagePassing` layer where each message is processed through a parameterised quantum circuit instead of a classical MLP. The circuit uses a Hardware-Efficient Ansatz (HEA) with alternating brick-wall CNOT entanglement, chosen for its resistance to barren plateaus (shallow depth, local observables, nearest-neighbour entanglement only).

### Data Re-uploading Quantum Circuit
The quantum circuit applies multiple re-uploading blocks (input encoding + HEA) in sequence, increasing expressivity without adding extra qubit depth.

### GRU Temporal Encoder
A GRU processes the raw price sequence for each stock before graph propagation, capturing short-term momentum. Its hidden state is concatenated with FinBERT sentiment embeddings to form the initial node features.

### FinBERT Sentiment Analysis
Financial news headlines are scored with [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert), producing per-ticker positive/negative/neutral probability vectors that are fused directly into the node feature space.

### Dynamic Graph Construction (Zero Data Leakage)
Stock correlation graphs are built from a rolling window of past prices only, so no future information leaks into the edge weights at any time step.

### Walk-Forward Cross-Validation
Training uses an expanding-window walk-forward scheme across multiple folds, faithfully replicating live deployment conditions and preventing lookahead bias.

### Sharpe Ratio Evaluation
Beyond MSE loss, each epoch is evaluated by the annualised Sharpe ratio of the prediction errors, giving a financially meaningful performance signal.

### Next-Day Price Prediction
After training, the last-fold model produces next-day price forecasts for AAPL, MSFT, and GOOGL, reported in USD alongside the last observed price.

### Visualisation Dashboard
Three figures are saved automatically to `plots/`:
- **Training Dashboard** — per-fold MSE loss and Sharpe ratio across epochs, full-period Pearson correlation heatmap, and FinBERT sentiment bar chart.
- **Actual vs Predicted** — walk-forward validation predictions overlaid on historical prices for each ticker, with the next-day forecast marked.
- **Rolling Correlation Evolution** — snapshots of the 30-day rolling correlation matrix at days 30, 60, and 99.

## Report

[View full project report (PDF)](report.pdf)

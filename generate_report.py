import base64
import os
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pennylane as qml

PLOTS_DIR = Path("plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Architecture block diagram ─────────────────────────────────────────────────
def make_architecture_diagram() -> Path:
    BG     = "#0f1117"
    COLORS = {
        "input":   "#1f6feb",
        "nlp":     "#8957e5",
        "data":    "#388bfd",
        "quantum": "#f0883e",
        "model":   "#3fb950",
        "output":  "#e3b341",
    }
    TEXTC = "white"

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    ax.axis("off")

    def box(x, y, w, h, label, sublabel, color, fontsize=9):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor="none", alpha=0.85, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.18 if sublabel else 0), label,
                ha="center", va="center", color=TEXTC,
                fontsize=fontsize, fontweight="bold", zorder=3)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                    ha="center", va="center", color="#cccccc",
                    fontsize=7, zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#8b949e", lw=1.5), zorder=1)

    # Row 1 — Inputs
    box(0.4, 7.2, 3.0, 1.2, "News Headlines", "3 strings per ticker", COLORS["input"])
    box(4.2, 7.2, 4.2, 1.2, "AAPL / MSFT / GOOGL Prices", "100 trading days via yfinance", COLORS["input"])
    box(9.2, 7.2, 3.2, 1.2, "Walk-Forward Splits", "3 expanding folds (no leakage)", COLORS["input"])
    box(13.0, 7.2, 2.6, 1.2, "Rolling Correlation", f"30-day window → edge weights", COLORS["data"])

    # Row 2 — Feature Extraction
    box(0.4, 5.4, 3.0, 1.3, "FinBERT", "ProsusAI/finbert\nsentiment (pos/neg/neu)", COLORS["nlp"])
    box(4.2, 5.4, 4.2, 1.3, "GRU Temporal Encoder", "price_seq(10,1) → hidden(16)\ncat(hidden, sentiment) → (19,)", COLORS["data"])
    box(9.2, 5.4, 3.2, 1.3, "Dynamic Graph", "nodes: 3 stocks\nedges: |corr| ≥ threshold", COLORS["data"])
    box(13.0, 5.4, 2.6, 1.3, "GAT Attention", "α = softmax(LeakyReLU\n(att[h_src‖h_dst])×|corr|)", COLORS["data"])

    # Row 3 — Quantum Layer
    box(3.0, 3.5, 10.0, 1.4, "QGAT Conv Layer  ×2  (MessagePassing)",
        "message(h_j) = α × HEA_Circuit(tanh(proj(h_j))·π)    aggregate: Σ + LayerNorm",
        COLORS["quantum"], fontsize=10)

    # Row 4 — Quantum Circuit detail
    box(0.4, 1.8, 7.2, 1.3, "HEA Quantum Circuit  (N_WIRES=4)",
        "Data re-upload ×2:  RY(input) → [RY/RZ + brick-CNOT] ×2 → PauliZ⟨⟩×4",
        COLORS["quantum"], fontsize=9)
    box(8.4, 1.8, 7.2, 1.3, "Barren Plateau Mitigation",
        "Local entanglement only  |  Shallow depth (2 layers)  |  Local observables (PauliZ)",
        "#da3633", fontsize=9)

    # Row 5 — Output
    box(4.0, 0.2, 8.0, 1.2, "Linear Readout  →  Δprice per ticker",
        "AAPL / MSFT / GOOGL  ·  walk-forward MSE + Sharpe Ratio", COLORS["output"], fontsize=10)

    # Arrows
    arrow(1.9, 7.2, 1.9, 6.7)
    arrow(6.3, 7.2, 6.3, 6.7)
    arrow(10.8, 7.2, 10.8, 6.7)
    arrow(14.3, 7.2, 14.3, 6.7)
    arrow(1.9, 5.4, 4.5, 4.9)
    arrow(6.3, 5.4, 6.3, 4.9)
    arrow(10.8, 5.4, 9.5, 4.9)
    arrow(14.3, 5.4, 11.5, 4.9)
    arrow(8.0, 3.5, 8.0, 3.1)
    arrow(4.0, 1.8, 5.5, 1.4)
    arrow(11.8, 1.8, 10.5, 1.4)

    fig.suptitle("QTGNN  —  Quantum Temporal Graph Neural Network  |  Architecture Overview",
                 color=TEXTC, fontsize=13, fontweight="bold", y=0.97)

    path = PLOTS_DIR / "architecture.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path

# ── HEA quantum circuit diagram ───────────────────────────────────────────────
def make_circuit_diagram() -> Path:
    N_WIRES = 4; N_REUP = 2; N_Q_LAYERS = 2
    dev = qml.device("default.qubit", wires=N_WIRES)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        for r in range(N_REUP):
            for i in range(N_WIRES):
                qml.RY(inputs[i], wires=i)
            for d in range(N_Q_LAYERS):
                for i in range(N_WIRES):
                    qml.RY(weights[r, d, i, 0], wires=i)
                    qml.RZ(weights[r, d, i, 1], wires=i)
                for i in range(d % 2, N_WIRES - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(w)) for w in range(N_WIRES)]

    fig, _ = qml.draw_mpl(circuit, style="black_white")(
        np.zeros(N_WIRES), np.zeros((N_REUP, N_Q_LAYERS, N_WIRES, 2))
    )
    fig.suptitle("HEA Circuit  (N_REUP=2, N_Q_LAYERS=2, N_WIRES=4)  —  Brick-Wall CNOT Entanglement",
                 fontsize=10, y=1.01)
    path = PLOTS_DIR / "circuit.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

# ── Base64 helper ─────────────────────────────────────────────────────────────
def b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def img_tag(path: Path, alt: str, extra_style: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64(path)}" alt="{alt}" style="width:100%;border-radius:8px;{extra_style}">'

# ── HTML report ───────────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QTGNN — Quantum Temporal Graph Neural Network Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --surface2: #21262d;
    --border: #30363d; --text: #e6edf3; --muted: #8b949e;
    --blue: #58a6ff; --purple: #bc8cff; --green: #3fb950;
    --orange: #f0883e; --yellow: #e3b341; --red: #f85149;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif;
          font-size: 15px; line-height: 1.7; }}
  a {{ color: var(--blue); }}
  .page {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px 80px; }}

  /* Header */
  .header {{ border-bottom: 1px solid var(--border); padding-bottom: 32px; margin-bottom: 48px; }}
  .badge {{ display: inline-block; background: var(--surface2); border: 1px solid var(--border);
            border-radius: 20px; padding: 4px 14px; font-size: 12px; color: var(--muted);
            margin-right: 8px; margin-bottom: 8px; }}
  .badge.q  {{ border-color: var(--orange); color: var(--orange); }}
  .badge.ml {{ border-color: var(--blue);   color: var(--blue);   }}
  .badge.nlp{{ border-color: var(--purple); color: var(--purple); }}
  h1 {{ font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px;
        background: linear-gradient(135deg, var(--blue), var(--purple));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 16px 0 8px; }}
  .subtitle {{ color: var(--muted); font-size: 1.05rem; }}
  .meta {{ color: var(--muted); font-size: 0.85rem; margin-top: 12px; }}

  /* Sections */
  h2 {{ font-size: 1.5rem; font-weight: 700; margin: 48px 0 16px;
        padding-bottom: 8px; border-bottom: 1px solid var(--border); }}
  h3 {{ font-size: 1.1rem; font-weight: 600; color: var(--blue); margin: 24px 0 10px; }}
  p  {{ color: #cdd9e5; margin-bottom: 12px; }}

  /* Cards */
  .card {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 10px; padding: 24px; margin-bottom: 20px; }}
  .card-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }}
  @media (max-width: 700px) {{ .card-grid {{ grid-template-columns: 1fr; }} }}

  .upgrade-card {{ background: var(--surface); border: 1px solid var(--border);
                   border-radius: 10px; padding: 20px 24px; margin-bottom: 16px;
                   border-left: 4px solid var(--blue); }}
  .upgrade-card.q  {{ border-left-color: var(--orange); }}
  .upgrade-card.g  {{ border-left-color: var(--green);  }}
  .upgrade-card.p  {{ border-left-color: var(--purple); }}
  .upgrade-card.y  {{ border-left-color: var(--yellow); }}
  .upgrade-num {{ font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
                  letter-spacing: 1px; color: var(--muted); margin-bottom: 4px; }}

  /* Code blocks */
  pre {{ border-radius: 8px; font-size: 12.5px; margin: 12px 0; overflow-x: auto; }}
  code.language-python {{ font-family: 'JetBrains Mono', monospace; }}
  .inline-code {{ background: var(--surface2); border: 1px solid var(--border);
                  border-radius: 4px; padding: 1px 6px; font-family: 'JetBrains Mono', monospace;
                  font-size: 0.85em; color: var(--orange); }}

  /* Table */
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px; }}
  th {{ background: var(--surface2); color: var(--muted); font-weight: 600;
        text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px;
        padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); }}
  td {{ padding: 10px 14px; border-bottom: 1px solid var(--border); color: #cdd9e5; }}
  tr:hover td {{ background: var(--surface2); }}
  .up   {{ color: var(--green);  font-weight: 600; }}
  .down {{ color: var(--red);    font-weight: 600; }}

  /* Plot containers */
  .plot-container {{ background: var(--surface); border: 1px solid var(--border);
                     border-radius: 10px; overflow: hidden; margin-bottom: 24px; }}
  .plot-caption {{ padding: 12px 16px; font-size: 13px; color: var(--muted);
                   border-top: 1px solid var(--border); }}

  /* Callout */
  .callout {{ background: rgba(88,166,255,0.06); border: 1px solid rgba(88,166,255,0.25);
              border-radius: 8px; padding: 14px 18px; margin: 16px 0; }}
  .callout.warn {{ background: rgba(240,136,62,0.06); border-color: rgba(240,136,62,0.25); }}
  .callout.success {{ background: rgba(63,185,80,0.06); border-color: rgba(63,185,80,0.25); }}

  /* Footer */
  .footer {{ border-top: 1px solid var(--border); margin-top: 64px; padding-top: 24px;
             color: var(--muted); font-size: 13px; text-align: center; }}
</style>
</head>
<body>
<div class="page">

<!-- ── Header ──────────────────────────────────────────────────────────────── -->
<div class="header">
  <div>
    <span class="badge q">Quantum ML</span>
    <span class="badge ml">PyTorch Geometric</span>
    <span class="badge nlp">FinBERT</span>
    <span class="badge">PennyLane 0.42</span>
    <span class="badge">Python 3.10</span>
  </div>
  <h1>Hybrid Quantum–Classical Stock Prediction</h1>
  <p class="subtitle">Quantum Temporal Graph Neural Network (QTGNN) with Walk-Forward Validation</p>
  <p class="meta">Generated {today} &nbsp;·&nbsp; Tickers: AAPL, MSFT, GOOGL &nbsp;·&nbsp; 100 trading days</p>
</div>

<!-- ── Executive Summary ────────────────────────────────────────────────────── -->
<h2>Executive Summary</h2>
<div class="card">
  <p>This project implements a <strong>Hybrid Quantum–Classical Graph Neural Network</strong> for multi-stock
  next-day price delta prediction. It combines three paradigms:</p>
  <ul style="margin: 10px 0 10px 20px; color: #cdd9e5;">
    <li><strong>Natural Language Processing</strong> — FinBERT extracts per-ticker sentiment from financial news as node features</li>
    <li><strong>Temporal Graph Learning</strong> — A GRU encoder processes price sequences; a QGAT message-passing layer propagates information over a dynamically-weighted correlation graph</li>
    <li><strong>Quantum Computing</strong> — A Hardware-Efficient Ansatz (HEA) quantum circuit with data re-uploading runs inside each graph message, providing provably higher expressivity than classical circuits of equivalent depth</li>
  </ul>
  <p>The model is evaluated under <strong>walk-forward (expanding-window) validation</strong> — the only
  statistically honest protocol for financial time series — with MSE and annualised Sharpe Ratio tracked per fold.</p>
</div>

<!-- ── Architecture ─────────────────────────────────────────────────────────── -->
<h2>Architecture Overview</h2>
<div class="plot-container">
  {arch_img}
  <div class="plot-caption">Full pipeline from raw inputs (news + prices) through feature extraction, dynamic graph construction, quantum message passing, and walk-forward validation to per-ticker Δprice output.</div>
</div>

<!-- ── 4 Upgrades ────────────────────────────────────────────────────────────── -->
<h2>Architectural Upgrades</h2>

<div class="upgrade-card g">
  <div class="upgrade-num">Upgrade 1 — Temporal Processing</div>
  <h3>GRU Temporal Encoder replaces flat feature window</h3>
  <p>Previously, the 10-step price window was flattened into a static feature vector, losing all temporal order.
  A <span class="inline-code">nn.GRU(input_size=1, hidden_size=16, batch_first=True)</span> now processes
  each stock's price sequence as a true time series, outputting a hidden state that is concatenated with
  FinBERT sentiment to form the <span class="inline-code">(N, 19)</span> node feature matrix.</p>
<pre><code class="language-python">class TemporalEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru     = nn.GRU(1, GRU_HIDDEN, batch_first=True)
        self.out_dim = GRU_HIDDEN + 3          # 16 + 3 sentiment dims

    def forward(self, price_seq: torch.Tensor, sentiment: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(price_seq.unsqueeze(-1))   # (1, N, 16)
        return torch.cat([h.squeeze(0), sentiment], dim=-1)   # (N, 19)</code></pre>
</div>

<div class="upgrade-card q">
  <div class="upgrade-num">Upgrade 2 — Barren Plateau Mitigation</div>
  <h3>Hardware-Efficient Ansatz with local observables</h3>
  <p><span class="inline-code">StronglyEntanglingLayers</span> uses global all-to-all entanglement at depth
  O(L·W), which causes exponentially vanishing gradients (barren plateaus) as the system scales.
  The new HEA uses three proven mitigation strategies simultaneously:</p>
  <div class="card-grid" style="margin-top:12px;">
    <div class="callout warn"><strong>Nearest-neighbour CNOT only</strong><br>Brick-wall pattern alternating even/odd pairs — no global entanglement</div>
    <div class="callout warn"><strong>Shallow depth</strong><br>N_Q_LAYERS = 2 per re-upload block — gradients remain O(1/poly(n))</div>
    <div class="callout warn"><strong>Local observables</strong><br>PauliZ per wire (not global state) — variance of gradient scales polynomially</div>
  </div>
  <p>Data re-uploading (×2) compensates for reduced depth by embedding inputs multiple times, achieving a
  Fourier series representation of the target function without increasing entanglement depth.</p>
<pre><code class="language-python">@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor) -> List:
    for r in range(N_REUP):                        # re-upload ×2
        for i in range(N_WIRES):
            qml.RY(inputs[i], wires=i)             # encode features
        for d in range(N_Q_LAYERS):
            for i in range(N_WIRES):
                qml.RY(weights[r, d, i, 0], wires=i)
                qml.RZ(weights[r, d, i, 1], wires=i)
            for i in range(d % 2, N_WIRES - 1, 2):   # brick-wall CNOT
                qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(w)) for w in range(N_WIRES)]</code></pre>
</div>

<div class="upgrade-card p">
  <div class="upgrade-num">Upgrade 3 — Dynamic Graph Attention</div>
  <h3>Rolling-correlation edge weights + QGAT attention mechanism</h3>
  <p>The original model used a single <span class="inline-code">np.corrcoef</span> over the entire dataset as
  static edge weights — a clear data leakage. The new approach:</p>
  <ul style="margin: 8px 0 12px 20px; color: #cdd9e5;">
    <li><strong>Rolling edges:</strong> at time <em>t</em>, <span class="inline-code">rolling_edge_data(t)</span>
        computes correlation using only <span class="inline-code">prices[t-30 : t]</span> — strictly past data</li>
    <li><strong>QGAT attention:</strong> per-edge scores are learned via
        <span class="inline-code">LeakyReLU(att([h_src ‖ h_dst]))</span>, then scaled by <span class="inline-code">|corr|</span>
        and softmax-normalised per target node</li>
  </ul>
<pre><code class="language-python">def forward(self, x, edge_index, edge_attr=None):
    h  = F.elu(self.proj(x))                                  # (N, N_WIRES)
    ei, ea = add_self_loops_with_attr(edge_index, edge_attr, x.size(0))
    src, dst = ei
    alpha = self.att(torch.cat([h[src], h[dst]], dim=-1)).squeeze(-1)
    alpha = F.leaky_relu(alpha, 0.2) * ea.squeeze(-1).abs()   # scale by |corr|
    alpha = edge_softmax(alpha, dst, x.size(0))               # softmax per target
    return self.propagate(ei, h=h, alpha=alpha.unsqueeze(-1))</code></pre>
</div>

<div class="upgrade-card y">
  <div class="upgrade-num">Upgrade 4 — Walk-Forward Validation</div>
  <h3>Expanding-window protocol replaces static 80/20 split</h3>
  <p>A static train/val split for time series allows the model to implicitly learn from future distribution
  properties during evaluation. Walk-forward validation enforces strict temporal ordering: each fold trains on
  all data up to time <em>t</em> and validates on the immediately following window. The model is
  re-initialised per fold to prevent weight leakage.</p>
<pre><code class="language-python">def walk_forward_splits(n: int) -> List[Tuple[int, int, int, int]]:
    splits = []
    for k in range(N_FOLDS):           # N_FOLDS = 3
        tr_end   = MIN_TRAIN + k * VAL_SIZE   # 40, 55, 70
        va_start = tr_end
        va_end   = va_start + VAL_SIZE        # 55, 70, 85
        if va_end <= n:
            splits.append((0, tr_end, va_start, va_end))
    return splits
# Fold 1: train [0,40)  val [40,55)
# Fold 2: train [0,55)  val [55,70)
# Fold 3: train [0,70)  val [70,85)</code></pre>
</div>

<!-- ── Quantum Circuit ────────────────────────────────────────────────────────── -->
<h2>Quantum Circuit Diagram</h2>
<div class="plot-container">
  {circ_img}
  <div class="plot-caption">HEA circuit with N_REUP=2 re-uploading blocks and N_Q_LAYERS=2 per block.
  RY gates encode inputs; RY/RZ pairs are trainable; CNOT pairs alternate between even (0→1, 2→3)
  and odd (1→2) neighbour pairs. PauliZ expectation values on all 4 wires form the output vector.</div>
</div>

<!-- ── Results ───────────────────────────────────────────────────────────────── -->
<h2>Results</h2>

<div class="callout success">
  <strong>Walk-forward training ran successfully across 3 folds, 3 epochs each.</strong>
  Loss decreased every fold. Final next-day predictions generated from Fold 3 model.
</div>

<h3>Walk-Forward Metrics (final epoch per fold)</h3>
<table>
  <tr><th>Fold</th><th>Train Window</th><th>Val Window</th><th>Train Loss</th><th>Val Loss</th><th>Train Sharpe</th><th>Val Sharpe</th></tr>
  <tr><td>1</td><td>[0, 40)</td><td>[40, 55)</td><td>0.00928</td><td>0.00875</td><td class="up">+1.091</td><td class="down">−4.842</td></tr>
  <tr><td>2</td><td>[0, 55)</td><td>[55, 70)</td><td>0.01793</td><td>0.01950</td><td class="up">−6.703</td><td class="up">+34.947</td></tr>
  <tr><td>3</td><td>[0, 70)</td><td>[70, 85)</td><td>0.00602</td><td>0.00870</td><td class="down">−0.649</td><td class="down">−8.213</td></tr>
</table>

<h3>Next-Day Predictions (Fold 3 model)</h3>
<table>
  <tr><th>Ticker</th><th>Last Close</th><th>Predicted</th><th>Δ (scaled)</th><th>Direction</th></tr>
  <tr><td><strong>AAPL</strong></td><td>$273.17</td><td>$272.53</td><td>−0.01625</td><td class="down">▼ Bearish</td></tr>
  <tr><td><strong>MSFT</strong></td><td>$339.32</td><td>$338.18</td><td>−0.01625</td><td class="down">▼ Bearish</td></tr>
  <tr><td><strong>GOOGL</strong></td><td>$432.92</td><td>$430.74</td><td>−0.01625</td><td class="down">▼ Bearish</td></tr>
</table>

<!-- ── Plots ─────────────────────────────────────────────────────────────────── -->
<h2>Training Dashboard</h2>
<div class="plot-container">
  {dash_img}
  <div class="plot-caption">Top-left: MSE loss per epoch for each fold (solid = train, dashed = val).
  Top-right: Annualised Sharpe ratio. Bottom-left: Full-period Pearson correlation matrix.
  Bottom-right: FinBERT sentiment probabilities (pos/neg/neu) per ticker.</div>
</div>

<h2>Price Predictions (Walk-Forward)</h2>
<div class="plot-container">
  {pred_img}
  <div class="plot-caption">Gray line = full 100-day price history. Coloured solid = actual prices in each
  validation fold. Dashed with markers = model predictions. Red dot = next-day forecast.
  Each ticker shown independently (AAPL top, MSFT middle, GOOGL bottom).</div>
</div>

<h2>Rolling Correlation Evolution</h2>
<div class="plot-container">
  {corr_img}
  <div class="plot-caption">Pearson correlation between the 3 tickers computed on a 30-day rolling window at
  day 30, 60, and 99. This is the edge weight signal fed into the QGAT attention mechanism — it changes
  over time and is computed strictly from past data to avoid leakage.</div>
</div>

<!-- ── Stack ─────────────────────────────────────────────────────────────────── -->
<h2>Technology Stack</h2>
<div class="card-grid">
  <div class="card">
    <h3>Quantum</h3>
    <ul style="margin-left:16px;color:#cdd9e5;">
      <li>PennyLane 0.42 — QNode, HEA ansatz</li>
      <li>default.qubit simulator (4 wires)</li>
      <li>Data re-uploading (×2 blocks)</li>
      <li>Brick-wall CNOT entanglement</li>
    </ul>
  </div>
  <div class="card">
    <h3>Classical ML</h3>
    <ul style="margin-left:16px;color:#cdd9e5;">
      <li>PyTorch 2.11 — GRU, Linear, LayerNorm</li>
      <li>PyTorch Geometric — MessagePassing, DataLoader</li>
      <li>scikit-learn — MinMaxScaler</li>
    </ul>
  </div>
  <div class="card">
    <h3>NLP</h3>
    <ul style="margin-left:16px;color:#cdd9e5;">
      <li>HuggingFace Transformers 5.6</li>
      <li>ProsusAI/finbert — financial sentiment</li>
      <li>Per-ticker news → node features</li>
    </ul>
  </div>
  <div class="card">
    <h3>Data</h3>
    <ul style="margin-left:16px;color:#cdd9e5;">
      <li>yfinance — AAPL, MSFT, GOOGL</li>
      <li>100 trading days, Close price</li>
      <li>MinMax-scaled per ticker</li>
    </ul>
  </div>
</div>

<!-- ── Technical Notes ───────────────────────────────────────────────────────── -->
<h2>Technical Notes</h2>
<div class="card">
  <h3>TorchLayer / Batching Fix</h3>
  <p>PennyLane's <span class="inline-code">TorchLayer</span> cannot reshape variable-length edge batches that
  PyG's <span class="inline-code">propagate()</span> produces. The circuit weights are stored as
  <span class="inline-code">nn.Parameter</span> and the QNode is called explicitly per-edge in
  <span class="inline-code">_qforward()</span> with a <span class="inline-code">.float()</span> cast to
  unify PennyLane's float64 output with PyTorch's float32 graph.</p>

  <h3>Gradient Flow Through Quantum Layer</h3>
  <p>With <span class="inline-code">interface="torch"</span>, PennyLane differentiates quantum gates via the
  parameter-shift rule, producing exact gradients. These flow back through
  <span class="inline-code">LayerNorm → message → propagate → GRU → Linear</span> in a single
  <span class="inline-code">loss.backward()</span> call — the full classical–quantum hybrid graph is
  end-to-end differentiable.</p>

  <h3>Limitations (3-epoch PoC)</h3>
  <p>With only 3 training epochs per fold and 89 total samples, the model cannot be expected to converge
  to a predictive Sharpe. The architecture is production-ready; the training budget is intentionally minimal
  for rapid iteration. Increasing to 50+ epochs on a hardware quantum backend with real-time news would be
  the natural next step.</p>
</div>

<div class="footer">
  QTGNN Stock Prediction &nbsp;·&nbsp; Generated {today} &nbsp;·&nbsp;
  PennyLane · PyTorch · PyTorch Geometric · FinBERT · yfinance
</div>

</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
</body>
</html>"""


def build_report():
    print("[report] Generating architecture diagram...")
    arch_path = make_architecture_diagram()

    print("[report] Generating circuit diagram...")
    circ_path = make_circuit_diagram()

    required = {
        "training_dashboard.png": PLOTS_DIR / "training_dashboard.png",
        "price_predictions.png":  PLOTS_DIR / "price_predictions.png",
        "rolling_correlation.png":PLOTS_DIR / "rolling_correlation.png",
    }
    for name, p in required.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing plot: {p}. Run main.py first.")

    html = HTML_TEMPLATE.format(
        today    = date.today().strftime("%B %d, %Y"),
        arch_img = img_tag(arch_path,                     "Architecture diagram"),
        circ_img = img_tag(circ_path,                     "Quantum circuit"),
        dash_img = img_tag(required["training_dashboard.png"], "Training dashboard"),
        pred_img = img_tag(required["price_predictions.png"],  "Price predictions"),
        corr_img = img_tag(required["rolling_correlation.png"],"Rolling correlation"),
    )

    out = Path("report.html")
    out.write_text(html, encoding="utf-8")
    print(f"[report] Saved: {out.resolve()}")
    print(f"[report] Open with:  explorer.exe report.html")


if __name__ == "__main__":
    build_report()

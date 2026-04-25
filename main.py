import warnings
warnings.filterwarnings("ignore")

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
PLOTS_DIR  : str       = "plots"
TICKERS    : List[str] = ["AAPL", "MSFT", "GOOGL"]
SEQ_LEN    : int       = 10
N_WIRES    : int       = 4
N_Q_LAYERS : int       = 2     # HEA depth per re-upload block
N_REUP     : int       = 3     # data re-uploading repetitions
GRU_HIDDEN : int       = 16
CORR_WIN   : int       = 30    # rolling correlation window (days)
BATCH_SIZE : int       = 8
EPOCHS     : int       = 15
PATIENCE   : int       = 5      # early stopping patience (val loss)
LR         : float     = 1e-2
N_FOLDS    : int       = 3
VAL_SIZE   : int       = 15
MIN_TRAIN  : int       = 40

# ── NLP: FinBERT sentiment per ticker ────────────────────────────────────────
TICKER_NEWS: Dict[str, List[str]] = {
    "AAPL":  [
        "Apple reports record quarterly earnings beating analyst expectations.",
        "Apple unveils revolutionary AI chip sparking investor enthusiasm.",
    ],
    "MSFT":  [
        "Microsoft Azure growth accelerates amid cloud demand surge.",
        "Microsoft faces antitrust scrutiny over AI partnerships.",
    ],
    "GOOGL": [
        "Alphabet beats revenue estimates advertising rebound continues.",
        "Google faces new EU regulatory challenges over search dominance.",
    ],
}

log.info("Loading FinBERT...")
_tok  = AutoTokenizer.from_pretrained("ProsusAI/finbert")
_bert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
_bert.eval()

def get_sentiment(texts: List[str]) -> np.ndarray:
    enc = _tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = _bert(**enc).logits
    return torch.softmax(logits, dim=-1).mean(dim=0).numpy().astype(np.float32)

SENTIMENTS: Dict[str, np.ndarray] = {t: get_sentiment(n) for t, n in TICKER_NEWS.items()}
log.info("Sentiments: %s", {k: v.round(3).tolist() for k, v in SENTIMENTS.items()})

# ── Price data ────────────────────────────────────────────────────────────────
log.info("Downloading price data for %s...", TICKERS)
raw_df = yf.download(TICKERS, period="200d", progress=False)["Close"].dropna()
prices: np.ndarray = raw_df.values[-100:]  # (100, 3) unscaled

scalers: List[MinMaxScaler] = [MinMaxScaler() for _ in TICKERS]
prices_scaled: np.ndarray = np.column_stack([
    scalers[i].fit_transform(prices[:, i : i + 1]).flatten()
    for i in range(len(TICKERS))
]).astype(np.float32)  # (100, 3)

# ── Dynamic edge construction (rolling correlation, zero data leakage) ────────
def rolling_edge_data(t: int) -> Tuple[torch.Tensor, torch.Tensor]:
    end   = max(t, 1)
    start = max(0, end - CORR_WIN)
    seg   = prices_scaled[start:end]
    corr  = np.corrcoef(seg.T) if seg.shape[0] > 1 else np.eye(len(TICKERS))
    rows, cols, vals = [], [], []
    for i in range(len(TICKERS)):
        for j in range(len(TICKERS)):
            if i != j:
                rows.append(i); cols.append(j); vals.append(float(corr[i, j]))
    return (
        torch.tensor([rows, cols], dtype=torch.long),
        torch.tensor(vals, dtype=torch.float32).unsqueeze(1),
    )

# ── Dataset ───────────────────────────────────────────────────────────────────
def make_graph(t: int) -> Data:
    price_feats = prices_scaled[t : t + SEQ_LEN]                   # (SEQ_LEN, 3)
    sent_feats  = np.stack([SENTIMENTS[k] for k in TICKERS])       # (3, 3)
    x = torch.tensor(
        np.concatenate([price_feats.T, sent_feats], axis=1),       # (3, SEQ_LEN+3)
        dtype=torch.float32,
    )
    y = torch.tensor(
        prices_scaled[t + SEQ_LEN] - prices_scaled[t + SEQ_LEN - 1],
        dtype=torch.float32,
    )   # per-node next-day delta, shape (3,)
    ei, ea = rolling_edge_data(t)
    return Data(x=x, edge_index=ei, edge_attr=ea, y=y)

class StockGraphDataset(Dataset):
    def __init__(self, start: int, end: int) -> None:
        super().__init__()
        self._idx: List[int] = list(range(start, end))

    def len(self) -> int:
        return len(self._idx)

    def get(self, idx: int) -> Data:
        return make_graph(self._idx[idx])

# ── Walk-forward splits (expanding window) ────────────────────────────────────
def walk_forward_splits(n: int) -> List[Tuple[int, int, int, int]]:
    splits = []
    for k in range(N_FOLDS):
        tr_end   = MIN_TRAIN + k * VAL_SIZE
        va_start = tr_end
        va_end   = va_start + VAL_SIZE
        if va_end <= n:
            splits.append((0, tr_end, va_start, va_end))
    return splits

n_samples: int = prices_scaled.shape[0] - SEQ_LEN - 1
splits         = walk_forward_splits(n_samples)
log.info("Walk-forward: %d folds over %d samples", len(splits), n_samples)
for fi, (a, b, c, d) in enumerate(splits):
    log.info("  Fold %d: train [%d,%d)  val [%d,%d)", fi + 1, a, b, c, d)

# ── Quantum Circuit: HEA with brick-wall CNOT (barren-plateau resistant) ──────
#   - Nearest-neighbour CNOT only (local entanglement, shallow depth)
#   - Local PauliZ observables (not global)
#   - Data re-uploading (N_REUP=3) for Fourier expressivity
#   - Equivariant weights: RY/RZ shared across wires per layer (reduces params, improves generalisation)
#   - Rolling correlation encoded into wire 0 per re-upload block
dev = qml.device("default.qubit", wires=N_WIRES)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor, corr: torch.Tensor) -> List:
    for r in range(N_REUP):
        for i in range(N_WIRES):
            qml.RY(inputs[i], wires=i)
        qml.RY(corr, wires=0)          # encode rolling correlation into wire 0
        for d in range(N_Q_LAYERS):
            for i in range(N_WIRES):
                qml.RY(weights[r, d, 0], wires=i)   # equivariant: shared across wires
                qml.RZ(weights[r, d, 1], wires=i)
            for i in range(d % 2, N_WIRES - 1, 2):   # alternating brick-wall
                qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(w)) for w in range(N_WIRES)]

Q_WEIGHT_SHAPE: Tuple[int, ...] = (N_REUP, N_Q_LAYERS, 2)   # equivariant: shared per layer

# ── Graph attention helpers ───────────────────────────────────────────────────
def add_self_loops_with_attr(
    edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    sl     = torch.arange(num_nodes, device=edge_index.device)
    sl_ei  = torch.stack([sl, sl])
    sl_ea  = torch.ones(num_nodes, 1, device=edge_attr.device)
    return torch.cat([edge_index, sl_ei], dim=1), torch.cat([edge_attr, sl_ea])

def edge_softmax(
    alpha: torch.Tensor, dst: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    alpha  = alpha - alpha.detach().max()          # global stability shift
    a      = alpha.exp()
    dsum   = torch.zeros(num_nodes, device=a.device)
    dsum.scatter_add_(0, dst, a)
    return a / (dsum[dst] + 1e-8)

# ── Temporal Encoder: GRU processes the price window sequentially ─────────────
class TemporalEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru     = nn.GRU(1, GRU_HIDDEN, batch_first=True)
        self.out_dim = GRU_HIDDEN + 3

    def forward(self, price_seq: torch.Tensor, sentiment: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(price_seq.unsqueeze(-1))    # (1, N, GRU_HIDDEN)
        return torch.cat([h.squeeze(0), sentiment], dim=-1)   # (N, out_dim)

# ── Quantum GAT Convolutional Layer ───────────────────────────────────────────
#   TorchLayer doesn't support PyG's variable-sized edge batches, so we own
#   the quantum weights as nn.Parameter and call the QNode explicitly per edge.
class QGATConv(MessagePassing):
    def __init__(self, in_channels: int) -> None:
        super().__init__(aggr="add", node_dim=0)
        self.proj      = nn.Linear(in_channels, N_WIRES)
        self.att       = nn.Linear(2 * N_WIRES, 1, bias=False)
        self.q_weights = nn.Parameter(
            torch.empty(*Q_WEIGHT_SHAPE).uniform_(-np.pi, np.pi)
        )
        self.norm = nn.LayerNorm(N_WIRES)

    def _qforward(self, x: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            torch.stack(quantum_circuit(x[i], self.q_weights, corr[i]))
            for i in range(x.size(0))
        ]).float()  # (E, N_WIRES) — cast float64 PL output → float32

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h  = F.elu(self.proj(x))
        N  = x.size(0)
        ei, ea = add_self_loops_with_attr(edge_index, edge_attr, N)
        src, dst = ei
        alpha = self.att(torch.cat([h[src], h[dst]], dim=-1)).squeeze(-1)
        alpha = F.leaky_relu(alpha, 0.2) * ea.squeeze(-1).abs()
        alpha = edge_softmax(alpha, dst, N)
        return self.propagate(ei, h=h, alpha=alpha.unsqueeze(-1), ea=ea)

    def message(self, h_j: torch.Tensor, alpha: torch.Tensor, ea: torch.Tensor) -> torch.Tensor:
        corr_angle = torch.tanh(ea.squeeze(-1)) * torch.pi
        return alpha * self._qforward(torch.tanh(h_j) * torch.pi, corr_angle)

    def update(self, aggr_out: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.norm(aggr_out + h)   # residual: mitigates barren-plateau gradient vanishing

# ── Full QTGNN ────────────────────────────────────────────────────────────────
class QTGNNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc     = TemporalEncoder()
        self.conv1   = QGATConv(self.enc.out_dim)
        self.conv2   = QGATConv(N_WIRES)
        self.readout = nn.Linear(N_WIRES, 1)

    def forward(self, data: Data) -> torch.Tensor:
        h = self.enc(data.x[:, :SEQ_LEN], data.x[:, SEQ_LEN:])
        h = F.relu(self.conv1(h, data.edge_index, data.edge_attr))
        h = self.conv2(h, data.edge_index, data.edge_attr)
        return self.readout(h).squeeze(-1)

def make_model() -> Tuple[QTGNNModel, torch.optim.Adam]:
    m = QTGNNModel()
    q_params  = [p for n, p in m.named_parameters() if "q_weights" in n]
    cl_params = [p for n, p in m.named_parameters() if "q_weights" not in n]
    opt = torch.optim.Adam([
        {"params": cl_params, "lr": LR},
        {"params": q_params,  "lr": LR * 0.1},   # quantum params need smaller LR (QBGU)
    ])
    return m, opt

# ── Metrics ───────────────────────────────────────────────────────────────────
def sharpe(p: torch.Tensor, t: torch.Tensor) -> float:
    r = (p - t).detach().cpu().numpy()
    s = np.std(r)
    return float(np.mean(r) / (s + 1e-8) * np.sqrt(252)) if s > 1e-8 else 0.0

# ── Epoch runner ──────────────────────────────────────────────────────────────
def run_epoch(
    loader   : DataLoader,
    model    : QTGNNModel,
    optimizer: Optional[torch.optim.Adam],
    train    : bool,
) -> Tuple[float, float]:
    model.train(train)
    all_p, all_t = [], []
    total_loss   = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            p    = model(batch)
            t    = batch.y.view(-1)
            loss = F.mse_loss(p, t)
            if train and optimizer is not None:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            all_p.append(p.detach()); all_t.append(t.detach())
    pc, tc = torch.cat(all_p), torch.cat(all_t)
    return total_loss / max(len(loader), 1), sharpe(pc, tc)

# ── Walk-forward training ─────────────────────────────────────────────────────
all_history  : List[Dict[str, List[float]]] = []
all_val_preds: List[np.ndarray]             = []
all_val_true : List[np.ndarray]             = []
all_val_days : List[List[int]]              = []
last_model   : Optional[QTGNNModel]         = None

for fold_i, (tr_s, tr_e, va_s, va_e) in enumerate(splits):
    log.info("── Fold %d/%d  train[%d,%d) val[%d,%d) ──",
             fold_i + 1, len(splits), tr_s, tr_e, va_s, va_e)
    tr_ld = DataLoader(StockGraphDataset(tr_s, tr_e), batch_size=BATCH_SIZE, shuffle=True)
    va_ld = DataLoader(StockGraphDataset(va_s, va_e), batch_size=BATCH_SIZE, shuffle=False)

    model, optimizer = make_model()
    hist: Dict[str, List[float]] = {k: [] for k in ("tr_loss", "va_loss", "tr_sharpe", "va_sharpe")}

    best_val      = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(1, EPOCHS + 1):
        trl, trs = run_epoch(tr_ld, model, optimizer, train=True)
        val, vas = run_epoch(va_ld, model, None,      train=False)
        for key, v in zip(hist.keys(), [trl, val, trs, vas]):
            hist[key].append(v)
        log.info("  Epoch %d/%d | TrLoss=%.5f VaLoss=%.5f | TrSharpe=%+.3f VaSharpe=%+.3f",
                 epoch, EPOCHS, trl, val, trs, vas)

        if val < best_val:
            best_val   = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                log.info("  Early stopping at epoch %d (patience=%d)", epoch, PATIENCE)
                break

    if best_state is not None:
        model.load_state_dict(best_state)   # restore best weights for this fold

    all_history.append(hist)
    last_model = model

    # Collect val predictions per ticker
    va_ds = StockGraphDataset(va_s, va_e)
    fold_preds, fold_true, fold_days = [], [], []
    model.eval()
    for k in range(va_ds.len()):
        g  = va_ds.get(k)
        t_ = va_ds._idx[k]
        with torch.no_grad():
            dp = model(g).numpy()
        dt       = g.y.numpy()
        base     = prices_scaled[t_ + SEQ_LEN - 1]
        pred_usd = np.array([scalers[j].inverse_transform([[base[j] + dp[j]]])[0][0] for j in range(len(TICKERS))])
        true_usd = np.array([scalers[j].inverse_transform([[base[j] + dt[j]]])[0][0] for j in range(len(TICKERS))])
        fold_preds.append(pred_usd); fold_true.append(true_usd)
        fold_days.append(t_ + SEQ_LEN)
    all_val_preds.append(np.array(fold_preds))
    all_val_true.append(np.array(fold_true))
    all_val_days.append(fold_days)

# ── Next-day prediction (last fold model) ────────────────────────────────────
assert last_model is not None
last_model.eval()
last_g = make_graph(n_samples - 1)
with torch.no_grad():
    d_sc = last_model(last_g).numpy()

log.info("Next-day predictions:")
next_day: Dict[str, Tuple[float, float]] = {}
for i, ticker in enumerate(TICKERS):
    ls = float(prices_scaled[-1, i])
    lp = float(scalers[i].inverse_transform([[ls]])[0][0])
    pp = float(scalers[i].inverse_transform([[max(0.0, ls + d_sc[i])]])[0][0])
    next_day[ticker] = (lp, pp)
    log.info("  %-5s | Last $%.2f → Predicted $%.2f | Δscaled=%+.5f", ticker, lp, pp, d_sc[i])

# ── Plotting ──────────────────────────────────────────────────────────────────
os.makedirs(PLOTS_DIR, exist_ok=True)
STOCK_COLORS = ["#4C9BE8", "#E8934C", "#6ABF6A"]
FOLD_COLORS  = ["#7B68EE", "#FF8C69", "#20B2AA"]

# ── Figure 1: Walk-forward training dashboard ─────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(14, 9))
fig1.suptitle("QTGNN Walk-Forward Training Dashboard", fontsize=14, fontweight="bold")

ax = axes[0, 0]
for fi, hist in enumerate(all_history):
    ex = list(range(1, len(hist["tr_loss"]) + 1))   # actual epochs run (early stop aware)
    ax.plot(ex, hist["tr_loss"], color=FOLD_COLORS[fi], ls="-",  label=f"F{fi+1} Train")
    ax.plot(ex, hist["va_loss"], color=FOLD_COLORS[fi], ls="--", label=f"F{fi+1} Val")
ax.set_title("MSE Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

ax = axes[0, 1]
for fi, hist in enumerate(all_history):
    ex = list(range(1, len(hist["tr_sharpe"]) + 1))
    ax.plot(ex, hist["tr_sharpe"], color=FOLD_COLORS[fi], ls="-",  label=f"F{fi+1} Train")
    ax.plot(ex, hist["va_sharpe"], color=FOLD_COLORS[fi], ls="--", label=f"F{fi+1} Val")
ax.axhline(0, color="gray", lw=0.8, ls=":")
ax.set_title("Sharpe Ratio (annualised)"); ax.set_xlabel("Epoch"); ax.set_ylabel("Sharpe")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

ax = axes[1, 0]
full_corr = np.corrcoef(prices.T)
im = ax.imshow(full_corr, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_xticks(range(len(TICKERS))); ax.set_xticklabels(TICKERS)
ax.set_yticks(range(len(TICKERS))); ax.set_yticklabels(TICKERS)
for r in range(len(TICKERS)):
    for c in range(len(TICKERS)):
        ax.text(c, r, f"{full_corr[r,c]:.2f}", ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="white" if abs(full_corr[r, c]) > 0.7 else "black")
fig1.colorbar(im, ax=ax, fraction=0.046)
ax.set_title("Full-Period Pearson Correlation")

ax = axes[1, 1]
labels = ["Positive", "Negative", "Neutral"]
xb = np.arange(len(labels)); w = 0.25
for i, (ticker, color) in enumerate(zip(TICKERS, STOCK_COLORS)):
    ax.bar(xb + i * w, SENTIMENTS[ticker], w, label=ticker, color=color, alpha=0.85)
ax.set_xticks(xb + w); ax.set_xticklabels(labels)
ax.set_ylim(0, 1); ax.legend(); ax.grid(axis="y", alpha=0.3)
ax.set_title("FinBERT Sentiment per Ticker"); ax.set_ylabel("Probability")

fig1.tight_layout()
p1 = os.path.join(PLOTS_DIR, "training_dashboard.png")
fig1.savefig(p1, dpi=150, bbox_inches="tight")
log.info("Saved: %s", p1)

# ── Figure 2: Actual vs predicted per ticker (all folds overlaid) ─────────────
fig2, axes2 = plt.subplots(len(TICKERS), 1, figsize=(14, 4 * len(TICKERS)))
fig2.suptitle("QTGNN: Actual vs Predicted — Walk-Forward Validation", fontsize=13, fontweight="bold")
all_days = np.arange(len(prices))

for i, (ticker, sc) in enumerate(zip(TICKERS, STOCK_COLORS)):
    ax = axes2[i]
    ax.plot(all_days, prices[:, i], color="lightgray", lw=1.5, zorder=1, label="History")
    for fi in range(len(all_val_days)):
        fd = all_val_days[fi]; fc = FOLD_COLORS[fi]
        ax.plot(fd, all_val_true[fi][:, i],  color=fc, lw=2,   label=f"F{fi+1} Actual")
        ax.plot(fd, all_val_preds[fi][:, i], color=fc, lw=1.5, ls="--",
                marker="x", ms=4, label=f"F{fi+1} Pred")
    lp, pp = next_day[ticker]
    ax.scatter(len(prices), pp, color="red", s=90, zorder=5, label=f"Next-day ${pp:.2f}")
    ax.set_title(ticker); ax.set_ylabel("Price (USD)"); ax.set_xlabel("Day index")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

fig2.tight_layout()
p2 = os.path.join(PLOTS_DIR, "price_predictions.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
log.info("Saved: %s", p2)

# ── Figure 3: Rolling correlation heatmap evolution (3 snapshots) ─────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))
fig3.suptitle("Rolling Correlation Evolution (CORR_WIN=30d)", fontsize=12, fontweight="bold")
snap_days = [30, 60, 99]
for ax, snap in zip(axes3, snap_days):
    start = max(0, snap - CORR_WIN)
    seg   = prices_scaled[start:snap]
    corr  = np.corrcoef(seg.T) if seg.shape[0] > 1 else np.eye(len(TICKERS))
    im    = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(TICKERS))); ax.set_xticklabels(TICKERS)
    ax.set_yticks(range(len(TICKERS))); ax.set_yticklabels(TICKERS)
    for r in range(len(TICKERS)):
        for c in range(len(TICKERS)):
            ax.text(c, r, f"{corr[r,c]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(corr[r, c]) > 0.7 else "black")
    ax.set_title(f"Day {snap}")
    fig3.colorbar(im, ax=ax, fraction=0.046)

fig3.tight_layout()
p3 = os.path.join(PLOTS_DIR, "rolling_correlation.png")
fig3.savefig(p3, dpi=150, bbox_inches="tight")
log.info("Saved: %s", p3)
log.info("All plots in ./%s/", PLOTS_DIR)

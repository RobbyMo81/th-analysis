
"""
trade_analysis.py
-----------------

Reusable helpers to analyze realized trading results.

Expected INPUT (flexible schema):
- CSV or JSON with at least these columns (case-insensitive; underscore/dash tolerant):
    * timestamp (datetime-like)  -> e.g., "2025-09-05 10:15:23" or date + time columns
    * pnl (realized P/L per closed trade or per fill; positive/negative USD)
Optional columns (used if present):
    * symbol, side, qty, fees/commission, account, strategy, instrument
    * date, time (if timestamp isn’t provided)
    * proceeds, cost_basis (we will compute pnl = proceeds - cost_basis when pnl missing)

You can also provide a "column_map" dict to rename broker-specific columns to the expected names.
Example:
    column_map = {"Realized P/L": "pnl", "Trade Date": "date", "Symbol": "symbol"}

Main functions:
- load_trades(path, column_map=None): returns normalized pandas DataFrame
- compute_metrics(df): returns dict of key performance stats
- aggregate(df): returns dict of groupby DataFrames (daily, weekly, monthly, by_symbol)
- plot_equity_curve(df, out_path): saves cumulative PnL curve PNG
- plot_drawdown(df, out_path): saves drawdown curve PNG
- write_summary(df, metrics, out_dir): writes summary.json and summary.md

Notes:
- This module assumes REALIZED P/L is available (per-trade or per-fill). If only fills are given
  and realized P/L is missing, pass proceeds & cost_basis OR pre-aggregate to realized trades.
- For futures (/NQ, etc.), this works fine when the broker export includes per-trade P/L.
"""

from __future__ import annotations
from pathlib import Path
import json
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# --------- Utilities ---------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize columns to lowercase with simple replacements
    new_cols = {}
    for c in df.columns:
        k = c.strip().lower().replace(" ", "_").replace("-", "_")
        new_cols[c] = k
    df = df.rename(columns=new_cols)
    return df

def _coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _to_numeric_clean(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric, handling common currency formatting.

    This will strip $ and commas, convert parentheses to negatives, and fall back to
    pandas.to_numeric(errors='coerce').
    """
    # Work on strings to normalize formatting
    s = series.copy()
    # If already numeric, return numeric conversion
    try:
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors='coerce')
    except Exception:
        pass

    s = s.astype(str).str.strip()
    # Replace common currency markers
    # Convert parentheses to negative values: (123.45) -> -123.45
    s = s.str.replace(r"\((.*)\)", r"-\1", regex=True)
    # Remove currency symbols and spaces
    s = s.str.replace(r"[$£€\s]", "", regex=True)
    # Remove thousands separators
    s = s.str.replace(r",", "", regex=True)

    return pd.to_numeric(s, errors='coerce')

def load_trades(path: str | Path, column_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Load trades from CSV or JSON and normalize columns.
    Attempts to build a 'timestamp' column and a 'pnl' column.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError("Only .csv or .json supported")

    # Apply user remap first if provided
    if column_map:
        df = df.rename(columns=column_map)

    df = _normalize_columns(df)

    # Try to build timestamp
    ts = None
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        # Try date + time
        date_col = None
        for cand in ["date", "trade_date", "execution_date"]:
            if cand in df.columns:
                date_col = cand
                break
        time_col = None
        for cand in ["time", "trade_time", "execution_time"]:
            if cand in df.columns:
                time_col = cand
                break
        if date_col is not None and time_col is not None:
            ts = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
        elif date_col is not None:
            ts = pd.to_datetime(df[date_col], errors="coerce")
    if ts is None:
        raise ValueError("Could not infer timestamp. Provide a 'timestamp' column or (date + time).")
    df["timestamp"] = ts

    # Attempt to compute pnl
    pnl = None
    for cand in ["pnl", "realized_p_l", "realized_pl", "realized", "realized_pnl", "profit_loss", "amount", "net_amount"]:
        if cand in df.columns:
            pnl = _to_numeric_clean(df[cand])
            break
    if pnl is None:
        # Try proceeds - cost_basis
        proceeds = None
        for cand in ["proceeds", "total_proceeds"]:
            if cand in df.columns:
                proceeds = _to_numeric_clean(df[cand])
                break
        cost = None
        for cand in ["cost_basis", "total_cost_basis", "cost"]:
            if cand in df.columns:
                cost = _to_numeric_clean(df[cand])
                break
        if proceeds is not None and cost is not None:
            pnl = proceeds - cost
        else:
            raise ValueError("Could not infer realized P/L. Provide 'pnl' or 'proceeds' & 'cost_basis'.")

    df["pnl"] = pnl.fillna(0.0)

    # Optional helpful fields
    if "symbol" not in df.columns:
        df["symbol"] = None
    if "qty" not in df.columns and "quantity" in df.columns:
        df["qty"] = pd.to_numeric(df["quantity"], errors="coerce")
    elif "qty" not in df.columns:
        df["qty"] = np.nan

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

# --------- Metrics ---------

@dataclass
class Metrics:
    total_trades: int
    win_trades: int
    loss_trades: int
    win_rate: float
    total_pnl: float
    average_gain: float
    average_loss: float
    gain_loss_ratio: float
    expectancy: float
    max_drawdown: float
    max_drawdown_start: Optional[pd.Timestamp]
    max_drawdown_end: Optional[pd.Timestamp]
    longest_win_streak: int
    longest_loss_streak: int

def compute_metrics(df: pd.DataFrame) -> Metrics:
    # Treat each row as a realized outcome
    pnl = df["pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    total_trades = len(df)
    win_trades = (pnl > 0).sum()
    loss_trades = (pnl < 0).sum()
    win_rate = win_trades / total_trades if total_trades else 0.0
    total_pnl = pnl.sum()
    average_gain = wins.mean() if len(wins) else 0.0
    average_loss = losses.mean() if len(losses) else 0.0  # negative
    gain_loss_ratio = (average_gain / abs(average_loss)) if average_gain and average_loss else np.nan
    expectancy = pnl.mean() if total_trades else 0.0

    # Equity curve & drawdown
    equity = pnl.cumsum()
    roll_max = equity.cummax()
    dd = equity - roll_max
    max_dd = dd.min() if len(dd) else 0.0
    max_dd_end_idx = dd.idxmin() if len(dd) else None
    # Safely determine the start index of the drawdown. When the end index is 0
    # or the slice up to the end is empty, idxmax would raise on an empty sequence.
    max_dd_start_idx = None
    if max_dd_end_idx is not None:
        # coerce index to integer position when possible
        try:
            end_pos = int(max_dd_end_idx)
        except Exception:
            end_pos = None
        try:
            if end_pos is not None and end_pos > 0:
                max_dd_start_idx = roll_max.iloc[:end_pos].idxmax()
            else:
                max_dd_start_idx = None
        except Exception:
            max_dd_start_idx = None

    # Resolve timestamps safely
    dd_start = None
    dd_end = None
    try:
        if max_dd_start_idx is not None:
            dd_start = pd.to_datetime(df.loc[int(max_dd_start_idx), "timestamp"], errors='coerce')
    except Exception:
        dd_start = None
    try:
        if max_dd_end_idx is not None:
            dd_end = pd.to_datetime(df.loc[int(max_dd_end_idx), "timestamp"], errors='coerce')
    except Exception:
        dd_end = None

    # Streaks
    signs = np.sign(pnl)
    longest_win_streak = 0
    longest_loss_streak = 0
    cur_win = cur_loss = 0
    for s in signs:
        if s > 0:
            cur_win += 1
            longest_win_streak = max(longest_win_streak, cur_win)
            cur_loss = 0
        elif s < 0:
            cur_loss += 1
            longest_loss_streak = max(longest_loss_streak, cur_loss)
            cur_win = 0
        else:
            cur_win = cur_loss = 0

    return Metrics(
        total_trades=total_trades,
        win_trades=win_trades,
        loss_trades=loss_trades,
        win_rate=float(win_rate),
        total_pnl=float(total_pnl),
        average_gain=float(average_gain if not np.isnan(average_gain) else 0.0),
        average_loss=float(average_loss if not np.isnan(average_loss) else 0.0),
        gain_loss_ratio=float(gain_loss_ratio) if not np.isnan(gain_loss_ratio) else float("nan"),
        expectancy=float(expectancy),
        max_drawdown=float(max_dd),
        max_drawdown_start=dd_start,
        max_drawdown_end=dd_end,
        longest_win_streak=longest_win_streak,
        longest_loss_streak=longest_loss_streak,
    )

def aggregate(df: pd.DataFrame):
    out = {}
    d = df.copy()
    d["date"] = d["timestamp"].dt.date
    # Use vectorized Period -> start_time conversion which is NaT-safe
    d["week"] = d["timestamp"].dt.to_period("W").dt.start_time.dt.date
    d["month"] = d["timestamp"].dt.to_period("M").astype(str)
    out["daily"] = d.groupby("date")["pnl"].sum().reset_index(name="daily_pnl")
    out["weekly"] = d.groupby("week")["pnl"].sum().reset_index(name="weekly_pnl")
    out["monthly"] = d.groupby("month")["pnl"].sum().reset_index(name="monthly_pnl")
    out["by_symbol"] = d.groupby("symbol")["pnl"].sum().reset_index(name="symbol_pnl").sort_values("symbol_pnl", ascending=False)
    return out

# --------- Plots ---------

def plot_equity_curve(df: pd.DataFrame, out_path: str | Path) -> Path:
    equity = df["pnl"].astype(float).cumsum()
    fig = plt.figure(figsize=(10,4))
    plt.plot(df["timestamp"], equity)
    plt.title("Cumulative Realized P&L")
    plt.xlabel("Time")
    plt.ylabel("Cumulative P&L ($)")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def plot_drawdown(df: pd.DataFrame, out_path: str | Path) -> Path:
    equity = df["pnl"].astype(float).cumsum()
    roll_max = equity.cummax()
    dd = equity - roll_max
    fig = plt.figure(figsize=(10,3))
    plt.plot(df["timestamp"], dd)
    plt.title("Drawdown (from Equity High)")
    plt.xlabel("Time")
    plt.ylabel("Drawdown ($)")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

# --------- Summary Writers ---------

def write_summary(df: pd.DataFrame, metrics: Metrics, out_dir: str | Path) -> Tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSON
    m = metrics.__dict__.copy()
    # Convert timestamps and numpy scalars to Python-native types for JSON
    for k, v in list(m.items()):
        # timestamps
        if isinstance(v, pd.Timestamp):
            m[k] = v.isoformat()
            continue
        # numpy integer types
        if isinstance(v, (np.integer,)):
            m[k] = int(v)
            continue
        # numpy floating types
        if isinstance(v, (np.floating,)):
            try:
                if np.isnan(v):
                    m[k] = None
                else:
                    m[k] = float(v)
            except Exception:
                m[k] = float(v)
            continue
        # python floats/ints/bools -> leave, but guard NaN
        if isinstance(v, float) and np.isnan(v):
            m[k] = None

    json_path = out_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(m, f, indent=2)

    # Markdown
    md = []
    md.append("# Performance Summary")
    md.append("")
    md.append(f"- Total Trades: **{metrics.total_trades}**")
    md.append(f"- Win Rate: **{metrics.win_rate:.2%}**  (Wins: {metrics.win_trades} / Losses: {metrics.loss_trades})")
    md.append(f"- Total Realized P&L: **${metrics.total_pnl:,.2f}**")
    md.append(f"- Average Gain: **${metrics.average_gain:,.2f}**  | Average Loss: **${metrics.average_loss:,.2f}**")
    if not np.isnan(metrics.gain_loss_ratio):
        md.append(f"- Gain/Loss Ratio: **{metrics.gain_loss_ratio:.2f}**")
    md.append(f"- Expectancy per Trade: **${metrics.expectancy:,.2f}**")
    md.append(f"- Max Drawdown: **${metrics.max_drawdown:,.2f}**")
    if metrics.max_drawdown_start and metrics.max_drawdown_end:
        md.append(f"  - From **{metrics.max_drawdown_start.date()}** to **{metrics.max_drawdown_end.date()}**")
    md.append(f"- Longest Win Streak: **{metrics.longest_win_streak}**  | Longest Loss Streak: **{metrics.longest_loss_streak}**")
    md.append("")
    md_path = out_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))

    return json_path, md_path

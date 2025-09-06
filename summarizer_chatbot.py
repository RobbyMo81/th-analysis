#!/usr/bin/env python3
"""Small rule-based chatbot to summarize analysis results.

Usage:
  python summarizer_chatbot.py --summary mnt/data/analysis/summary.json --question "Give me a short summary"
  python summarizer_chatbot.py            # interactive REPL
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    # support two shapes: { "metrics": { ... } } or top-level metrics
    if isinstance(raw, dict) and "metrics" in raw:
        return raw
    if isinstance(raw, dict):
        # if the file looks like metrics at the top-level, wrap it
        metric_like_keys = {"total_trades", "total_pnl", "win_rate", "max_drawdown"}
        if any(k in raw for k in metric_like_keys):
            return {"metrics": raw}
    return {"metrics": raw}


def short_summary(data: dict[str, Any]) -> str:
    m = data.get("metrics", {})
    total = m.get("total_pnl")
    trades = m.get("total_trades")
    win_rate = m.get("win_rate")
    max_dd = m.get("max_drawdown")
    lines = [f"Trades: {trades}", f"Total PnL: {total}", f"Win rate: {win_rate}", f"Max drawdown: {max_dd}"]
    return " | ".join(str(x) for x in lines if x is not None)


def answer_question(q: str, data: dict[str, Any]) -> str:
    ql = q.lower().strip()
    m = data.get("metrics", {})
    # simple intent matching
    if "summary" in ql or "overview" in ql:
        return short_summary(data)
    if "pnl" in ql or "profit" in ql or "loss" in ql:
        return f"Total PnL: {m.get('total_pnl')} over {m.get('total_trades')} trades."
    if "win" in ql and "rate" in ql:
        return f"Win rate: {m.get('win_rate')} ({m.get('win_trades')} wins / {m.get('loss_trades')} losses)."
    if "drawdown" in ql or "max drawdown" in ql:
        return f"Max drawdown: {m.get('max_drawdown')}, from {m.get('max_drawdown_start')} to {m.get('max_drawdown_end')}"
    if "top" in ql and ("wins" in ql or "gains" in ql):
        return f"Average gain: {m.get('average_gain')}, average loss: {m.get('average_loss')}, gain/loss ratio: {m.get('gain_loss_ratio')}"
    # fallback: echo important numbers
    keys = ["total_trades", "win_trades", "loss_trades", "win_rate", "total_pnl", "max_drawdown"]
    found = {k: m.get(k) for k in keys if k in m}
    if found:
        parts = [f"{k}: {v}" for k, v in found.items()]
        return "I couldn't parse that question precisely. Here are core metrics: " + "; ".join(parts)
    return "Sorry, I don't know how to answer that. Try: 'summary', 'pnl', 'win rate', 'drawdown'."


def repl(data: dict[str, Any]) -> None:
    print("Summary chatbot — type a question (or 'quit' to exit). Examples: 'summary', 'pnl', 'win rate', 'drawdown'.")
    while True:
        try:
            q = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.strip().lower() in {"quit", "exit"}:
            break
        print(answer_question(q, data))


def main() -> None:
    p = argparse.ArgumentParser(description="Small summary chatbot for analysis results")
    p.add_argument("--summary", "-s", default="mnt/data/analysis/summary.json", help="path to summary.json")
    p.add_argument("--question", "-q", help="single question to answer (non-interactive)")
    args = p.parse_args()

    summary_path = Path(args.summary)
    data = load_summary(summary_path)

    if args.question:
        print(answer_question(args.question, data))
    else:
        repl(data)


if __name__ == "__main__":
    main()

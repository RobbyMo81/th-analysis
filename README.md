# th-analysis

Small trade analysis project. Run `run_analysis.py` to compute metrics and produce plots in `mnt/data/analysis`.

Chatbot summarizer

There is a small rule-based chatbot that reads `mnt/data/analysis/summary.json` and answers basic questions.

Usage:

```powershell
& .\.venv\Scripts\python.exe summarizer_chatbot.py --summary mnt/data/analysis/summary.json --question "Give me a short summary"
# or interactive REPL
& .\.venv\Scripts\python.exe summarizer_chatbot.py
```

Supported questions: summary, pnl, win rate, drawdown, average gain/loss.
# Project

Bootstrapped README.

# run_pipeline.py
# Master script — runs the full pipeline from data to trained model
# Run this once to set everything up.

import subprocess
import sys
import os

steps = [
    ("Step 1: Collect Data",       "python data/collect_data.py"),
    ("Step 2: Score Sentiment",    "python features/sentiment.py"),
    ("Step 3: Engineer Features",  "python features/engineer_features.py"),
    ("Step 4: Train LSTM Model",   "python models/train.py"),
]

print("\n" + "="*55)
print("  SentimentEdge — Full Training Pipeline")
print("="*55)

for label, cmd in steps:
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"  $ {cmd}")
    print(f"{'─'*55}\n")

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n  ERROR in '{label}'. Check output above.")
        sys.exit(1)

print("\n" + "="*55)
print("  Pipeline complete!")
print("  Start the API:  python api/serve.py")
print("  Open index.html in your browser")
print("="*55)
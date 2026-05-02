"""
=============================================================
RUN_ALL.PY — Master Pipeline Runner
=============================================================
PURPOSE:
    Run all 5 training stages in order:
        Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5

    After this script completes successfully, launch the
    Streamlit dashboard with:

        streamlit run stage6_dashboard.py

USAGE:
    python run_all.py

NOTE:
    - Stage 1 requires an internet connection to download
      data from Yahoo Finance.
    - Stages 4 and 5 are GPU-accelerated if TensorFlow
      detects a CUDA-compatible GPU; otherwise they run on CPU.
    - Total runtime: ~10-30 minutes depending on hardware.
=============================================================
"""

import sys
import time

def run_stage(name: str, fn):
    """Run a single stage function with timing and error handling."""
    print(f"\n{'='*60}")
    print(f"  RUNNING: {name}")
    print(f"{'='*60}")
    start = time.time()
    try:
        fn()
        elapsed = time.time() - start
        print(f"\n  ✓  {name} completed in {elapsed:.1f}s")
    except Exception as e:
        print(f"\n  ✗  {name} FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":

    # ── Stage 1: Data Collection ──────────────────────────
    from stage1_data_collection import collect_all_stocks
    run_stage("Stage 1: Data Collection", collect_all_stocks)

    # ── Stage 2: Preprocessing & Feature Engineering ──────
    from stage2_preprocessing import preprocess
    run_stage("Stage 2: Preprocessing & Feature Engineering", preprocess)

    # ── Stage 3: Labelling & Sequences ───────────────────
    from stage3_labelling_sequences import prepare_sequences
    run_stage("Stage 3: Labelling, Normalisation & Sequences",
              prepare_sequences)

    # ── Stage 4: Model Training ───────────────────────────
    from stage4_model_training import train_model
    run_stage("Stage 4: CNN-LSTM Model Training", train_model)

    # ── Stage 5: Evaluation ───────────────────────────────
    from stage5_evaluation import evaluate
    run_stage("Stage 5: Model Evaluation", evaluate)

    # ── Done ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ALL STAGES COMPLETE!")
    print("  To launch the dashboard, run:")
    print("      streamlit run stage6_dashboard.py")
    print(f"{'='*60}\n")

# TASR Reproducibility Package

This folder is a cleaned, public-safe export of the experiment code used for the TASR paper.

The project studies adaptive stopping for multi-round question answering. The core workflow is:

1. Run a retrieve-and-answer agent for up to 5 rounds per question.
2. Save a per-round parquet with answer quality, retrieval signals, confidence, and logit margins.
3. Calibrate and enrich those signals.
4. Replay stopping rules offline on the cached parquet files to measure F1 and average calls.

Most of the paper numbers can be reproduced from the bundled parquets in `results/` without making any new LLM calls.

## What This Package Contains

- `src/`: runnable Python scripts for trace generation, enrichment, replay, ablations, and plotting
- `data/`: bundled HotpotQA distractor splits, bundled 2Wiki splits, and the evaluation helper
- `results/`: cached experiment outputs and enriched parquets used for offline reproduction
- `figures/`: generated paper-facing figures
- `requirements.txt`: Python dependencies for this export
- `.env.example`: public-safe endpoint configuration template

This export intentionally excludes local clutter such as private `.env` files, logs, caches, indexes, virtual environments, and day-to-day working notes.

## How The Project Works

The codebase breaks into four layers:

1. Data and retrieval setup. The agent reads a HotpotQA-style JSON file and retrieves evidence passages round by round.
2. Trace generation. `run_*` scripts produce one row per `(question, round)` with answer quality, retrieval scores, overlap features, confidence, and optionally logit margins.
3. Signal enrichment. `signals.py`, `enrich_lp.py`, and `enrich_v2_signals.py` add calibrated confidence, calibrated margin, and robust overlap and z-score features.
4. Offline evaluation. `exp_*`, `replay_*`, `subset_ablation_*`, and `plot_*` scripts replay stopping rules on the saved parquets to produce tables, confidence intervals, and figures.

Most of the heavy cost is in trace generation. Once the enriched parquets exist, almost everything else is pandas, evaluation logic, and bootstrap code.

## Repository Layout

- `src/run_*`: generate traces or run multi-stage pipelines
- `src/enrich_*` and `src/signals.py`: add derived and calibrated signals to raw traces
- `src/exp_*`: analysis scripts that emit tables or JSON summaries
- `src/replay_*`: replay locked stopping rules on existing parquets
- `src/plot_*` and `src/*pareto*`: generate figures from saved results
- `src/subset_ablation_lp_dnf.py`: exhaustive DNF stopping-rule sweep over the enriched logprob parquet
- `results/`: the main offline replay surface; most scripts read from here and write back here
- `figures/`: figure outputs produced by plotting scripts

## Setup

Create a virtual environment, install dependencies, and copy the environment template:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` and fill in the endpoint URLs, model names, and API keys you want to use.

## Configuration

The scripts use an OpenAI-compatible chat-completions API.

- `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`: default endpoint used by scripts that do not override the model
- `LLM_NO_THINKING_KWARG`: set this when your endpoint does not accept the extra thinking parameter
- `QWEN_*`, `DEVSTRAL_*`, `GEMMA_*`: optional per-model overrides used by the wrapper pipelines

The wrapper scripts for Devstral and Gemma set `LLM_*` from the model-specific variables before importing the client, so background or resumable runs do not depend on your shell state.

If you run Pyserini-based retrieval experiments, you also need Java 21. The retrieval scripts default to common Linux Java 21 paths, but you can override them with `JAVA_HOME` and `JVM_PATH`.

## Recommended First Run

The fastest way to verify the package is working is to rerun one offline analysis from the bundled parquets:

```bash
python src/exp_headline_table.py
```

That script recomputes the main headline comparison table from the enriched parquets already bundled in `results/`. It does not need any live LLM endpoint.

Other useful offline checks are:

```bash
python src/exp_robustness_checks.py
python src/plot_guardrail_frontier.py
python src/replay_wiki.py
python src/bootstrap_ci_opendomain.py
python src/replay_contriever.py
```

## Main Workflows

### 1. Offline Reproduction From Bundled Results

This is the workflow most users want. It reproduces the paper-facing tables and figures without new model calls.

Good entry points:

```bash
python src/exp_headline_table.py
python src/exp_robustness_checks.py
python src/subset_ablation_lp_dnf.py \
  --eval results/signal_log_lp_enriched.parquet \
  --tune results/signal_log_lp_enriched_tune.parquet \
  --out-csv results/subset_ablation_lp_dnf.csv \
  --out-json results/subset_ablation_lp_dnf.json
python src/plot_guardrail_frontier.py
python src/replay_wiki.py
python src/replay_contriever.py
```

Use this path if your goal is to verify headline numbers, compare stopping rules, regenerate confidence intervals, or remake figures from the saved traces.

The most important parquet columns for offline replay are `qid`, `round`, `current_answer`, `current_em`, `current_f1`, `llm_confidence`, `calibrated_conf`, `answer_token_margin`, `calibrated_logit_margin`, `overlap_signal`, and `gap`.

### 2. Regenerate HotpotQA Distractor Traces

The bundled `data/dev_tune.json` and `data/dev_eval.json` are the HotpotQA-style distractor splits used by the core experiments.

The most direct trace-generation script is:

```bash
python src/run_instrumented_with_logprobs.py --split both
```

This captures the k=5 retrieve-and-answer traces with logprobs and writes the logprob parquet files. It is the expensive stage and can take hours. The script checkpoints every 25 questions so interrupted runs can resume.

For full model-specific wrapper pipelines, use:

```bash
python src/run_devstral_pipeline.py
python src/run_gemma_pipeline.py
```

These wrappers run the whole staged workflow for the corresponding model family: closed-book baseline, plain k=5 traces, logprob traces, signal enrichment, offline evaluation, ablations, and figures. They are resumable and skip any stage whose output already exists.

The original Qwen HotpotQA cell is represented in this export mainly through the bundled results and the direct stage scripts rather than a single wrapper script. For most reproducibility use cases, the saved Qwen parquets in `results/` are the intended starting point.

### 3. Regenerate The 2WikiMultiHopQA Cells

The package already includes `data/2wiki_dev_tune.json` and `data/2wiki_dev_eval.json`. If you want to recreate them from the public dataset, run:

```bash
python src/prep_2wiki.py
```

Then run the full pipelines:

```bash
python src/run_qwen_2wiki_full.py
python src/run_devstral_2wiki_full.py
python src/run_gemma_2wiki_full.py
```

These scripts are multi-stage, resumable pipelines that write namespaced outputs under `results/<model>_2wiki/` and `figures/<model>_2wiki/`.

### 4. Open-Domain And Dense Retrieval Experiments

These experiments require assets that are not bundled in this export: Pyserini indexes, `ir_datasets` caches, and Java 21.

For Contriever dense retrieval on HotpotQA fullwiki, use commands like:

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export JVM_PATH=/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so

python src/run_contriever_experiment.py --model qwen --corpus fullwiki
python src/run_contriever_experiment.py --model devstral --corpus fullwiki
python src/run_contriever_experiment.py --model gemma --corpus fullwiki
```

After those parquets exist, replay the locked rules offline with:

```bash
python src/replay_contriever.py
```

For the BM25-based open-domain replay surface, use the existing parquets plus:

```bash
python src/replay_wiki.py
python src/bootstrap_ci_opendomain.py
```

## What Gets Written Where

The main outputs to watch are:

- `results/signal_log*.parquet`: raw or logprob per-round traces for the base HotpotQA distractor setting
- `results/*/signal_log*.parquet`: namespaced traces for model-specific or dataset-specific reruns
- `results/*_enriched*.parquet`: enriched traces with calibrated signals used by offline analyses
- `results/*.json` and `results/*.csv`: summary tables, confidence intervals, threshold sweeps, and ablation outputs
- `figures/` and `figures/*/`: saved plots and paper-facing figures

Most pipeline scripts are safe to rerun because they skip work when their target outputs already exist.

## Important Files

- `src/llm_client.py`: default client and environment-variable behavior
- `src/run_devstral_pipeline.py`: best example of the full staged workflow in one file
- `src/exp_headline_table.py`: fastest offline check for the main comparison table
- `src/subset_ablation_lp_dnf.py`: exhaustive rule sweep over the saved enriched traces

## Common Issues

- If a live pipeline fails immediately with an authentication or connection error, your `.env` is incomplete or points to a non-compatible endpoint.
- If a retrieval script fails when importing Pyserini or loading the JVM, check `JAVA_HOME`, `JVM_PATH`, and that Java 21 is installed.
- If a script complains about missing corpora or indexes, that asset is intentionally not bundled in this public export.
- If you only need the paper numbers, prefer the offline replay scripts over fresh pipeline runs. They are much faster and avoid external dependencies.

## Notes On This Export

- Hard-coded absolute paths from the working repo were replaced with repo-relative paths.
- Internal endpoint URLs, host labels, and local placeholder API keys were removed and replaced with env-driven configuration.
- Historical filename drift was normalized in this export. The main comparison table is produced by `src/exp_headline_table.py`, and the reranker comparison script is `src/exp_reranker_signal.py`.

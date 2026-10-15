# Adaptor

The adaptor is the part of LottoPipeline that looks at the running system
and decides whether a component should be reviewed later.

It does not pick lottery numbers. The draw data is the workload. The point
of the adaptor is to measure how the pipeline is built and how a training
batch performed, then name one component if that batch did not improve.

Later stages (not in this tree yet) are meant to search for a replacement
method, test it off to the side, and only keep it if the next real batch
is better. If it is not, the previous code stays. Stage 1 does none of that.
It only reads structure and performance and writes a report.

## Theory

A pipeline with several feature steps and a trainer will not stay useful
if a weak step is left in place and never checked.

The intended loop:

1. Run the pipeline on the latest draw.
2. Record what each file reads and writes, and what tensors actually appeared.
3. Score each fused feature block by training without it.
4. Compare this batch’s best validation AUC to recent batches.
5. If this batch did not beat that history, name one candidate
   (a step file or a quantum encoder) for later work.
6. Later: find a method that still fits the same inputs and outputs,
   test it in isolation, put it in only if the next batch improves,
   otherwise put the old file back.

One change per failed batch. Do not edit two files in the same cycle.
Do not delete a step because its score is low; replace the method inside
the file, or change encoder knobs, and keep the public shape the same.

Validation truth for the skip gate is peak `val_auc` for that date in
`lotto.db`, not training loss and not the ablation printout.

## Stages

| Stage | Role | In this repo |
| --- | --- | --- |
| 1 Info | Read files, record the run, score blocks, write a decision | Yes |
| 2 Search | Find a similar method for that candidate | Planned |
| 3 Simulate | Build and score candidates without writing `lotto.db` | Planned |
| 4 Isolated apply | Patch a copy, not main, until the gate passes | Planned |
| 5 Keep or revert | Flag, compare the next real peak, drop the backup if kept | Planned |

Stage 1 never edits `steps/` or `config/`.

## Stage 1 — Info layer

Four modules under `adaptor/`. `main.py` builds the pipeline, the observer,
and the assessor, runs the steps, then calls `run_optuna_bridge`.

### `assessment.py`

Walks `steps/` and `config/` as text. Does not import or run those files.

- Finds `get_data("…")` and `add_data("…")` string keys.
- Nested functions stay attached to the right def.
- File key is the basename (`clustering.py`), which matches
  `PIPE_TO_FILENAME` in the bridge.
- If a file defines `get_encoder_spec`, it is tagged `encoder`.
  The `output_key` string in that return dict is listed under outputs.
  That is not a fake `add_data` inside the quantum modules.

Quantum files still have empty `reads` unless they call `get_data`.
Deep learning is what writes `quantum_features` and `quantum_kernels`
onto the pipeline.

### `runtime_observer.py`

Hooks `pipeline.add_data` for the current run.

- `start_new_run` stores compact stats from the last run, then opens a new id.
- `get_run_summary` only reads. It does not move the previous-run memory.
- Each key gets type, shape, min/max/mean/std when the value is numeric.
- Vectors that look like a probability simplex also get entropy and L1 vs uniform.
- Deltas vs the previous run use entropy first, then std.
  Mean is not used for a simplex.

This is what the run produced, not what the source files declare.

### `ablation.py`

Leave-one-block-out retrain. No writes to `epochs` or `lotto.db`.

- Drops one column range from the fused train/val matrices.
- Trains a new head in memory.
- Score = baseline AUC − ablated AUC.
- Baseline is production val AUC from deep learning when that value is passed.
- Macro AUC skips labels that are all 0 or all 1 in val (Powerball 11–14
  until those numbers exist in history).
- `|score| < 0.002` is stored and printed as `0.0000`.
- Positive score: removing the block hurt. The block helped the baseline.
- Zero or negative: unused on this split, or the short retrain beat production.
- One candidate: lowest score among all blocks.
- `candidate_class` is `pipe` or `encoder`. Only one of
  `candidate_pipe` / `candidate_encoder` is set.

`model` on `compute_importance` is unused. The call site in deep learning
still passes it.

### `optuna_bridge.py`

Does not run Optuna. It reads `lotto.db` and the pipeline.

- Last six date batches: epoch count, average val AUC, peak val AUC.
- Skip the report if the latest peak is more than `0.005` above the median
  of the earlier peaks, or if there are fewer than two batches.
- If it does not skip: one candidate, one class, one ablation score,
  runtime summary for that name only.
- Assessment file dump only when the candidate is a pipe.
- Writes `adaptor_decision` on the pipeline. No source edit.

Actions in the record:

- `methodology_review` — pipe candidate
- `encoder_review` — encoder candidate
- `stack_review` — no candidate (all snapped scores were ~0)

## What Stage 1 does not do

- Search the web or call an API for a new method
- Rewrite a step or call `apply_encoder_spec`
- Treat a near-0.5 AUC as proof the model found structure in the draws
- Name two components for edit on the same failed batch
# Teekathon submission — VLM Fine Tuning

Fine-tunes a vision-language model (VLM) via Tinker to detect and crop
regions on scanned exam-paper pages (`cover_page`, `mcq_question`,
`oe_question`, `mcq_answer_key`, `oe_answer_key`), per the task spec in the
repo root `README.md`.

## Setup

```
uv sync
```

Requires `TINKER_API_KEY` in `.env` or the environment.

## Commands

- `uv run train --training-folder-path <folder> [--limit N] [--num-steps N] [--learning-rate F] [--run-id ID] [--save-sampler-weights]`
  Converts every paper in `<folder>` (each needs `gold.json`, `gold/`, `source.pdf`) into training examples via `build_training_examples_from_gold`, prints a real pre-spend cost estimate (text + image tokens, via `compute_expected_tokens`), asks for confirmation, then runs LoRA fine-tuning (`train_vlm`) and saves a checkpoint. `--run-id` resumes an existing run instead of starting a new one.
- `uv run main --image-path <pdf> --run-id <run_id> [--limit N]`
  Renders the PDF, runs inference with the checkpoint from `<run_id>` (sliding-window: overlapping page pairs, `merge_page_readings` reconciles the two independent reads of each interior page, `apply_continuation_chains` corrects labels on regions that span a page boundary), evaluates against `gold.json` if present (`evaluate_predictions_against_gold`, IoU >= 0.90), and writes `output/predictions.json`, `output/overlays/`, and `output/crops/<type>/`.
- `uv run spendings` — real Tinker spend so far (last 14 days), via the billing API, broken down by run.
- `uv run get-training-run --run-id <id>` — status of a specific training run.
- `uv run sample-runid --run-id <id>` — quick sample from a checkpoint without a full `main` pass.

## Data conversion / annotation

`build_training_examples_from_gold` (`src/main/main.py`) is the gold.json + crop -> training-example conversion step. It also implements two automated corrections found necessary during a structural audit of the supplied gold data (see "Gold data quality" below): duplicate-fragment removal (hash comparison, drops byte-identical repeated crops inside one region) and mismatched-region splitting (a region whose crops actually show multiple distinct questions is split into one training example per crop instead of one bundling several questions under one label).

## Training run

Trained on a pilot subset of `corrected_data/`, 8 of the 80 supplied papers (4 P5 Maths, 2 P5 Science, 2 P6 Science; no P6 Maths papers in this pilot), each manually/AI-corrected from the originally supplied `gold.json` before conversion. 184 training examples built from these 8 papers, 9 training steps, checkpoint saved at:

```
tinker://f1bea322-64be-58aa-a516-3b33ba7370f1:train:0/weights/checkpoint-final
```

(sampler weights also saved, for inference via `uv run main --run-id f1bea322-64be-58aa-a516-3b33ba7370f1:train:0`).

Real total spend and per-run breakdown: `uv run spendings` (or `total_usage.py`) gives the live number from Tinker's billing API. Last recorded snapshot before this pilot run's own spend: $2.449621 total training spend across all runs in the last 14 days (per-run breakdown available via the same command); this run's own estimated cost was $16.6684 (176601 text + 1396928 image tokens, over 9 steps), giving a combined total in the ~$19-20 range, well under the $80 cap. Re-run `uv run spendings` for the exact final figure before relying on this number.

Note: a real Tinker billing block (HTTP 402, "Access... is blocked due to billing status") interrupted an earlier training run mid-step. Confirm billing is in good standing at https://tinker.thinkingmachines.ai/billing/balance before assuming any command here will complete.

## Gold data quality

A structural audit across all 80 supplied papers found 231 regions with more than one crop fragment (`fragments` array length > 1 in `gold.json`). Manual/AI-assisted visual classification of every one of these 231 regions found: 226 bundle multiple distinct questions under one label (a sequential-labelling issue in the supplied data, not a training artifact), 5 are duplicate/near-duplicate crops, 0 are genuine continuations. `corrected_data/` papers use manually corrected annotations for the papers included in this pilot rather than the originally supplied ones.

## Known limitations

- Trained on 8 of 80 papers only, not the full dataset, due to time window.
- Sliding-window inference reads each interior page twice (as the second image of one page pair, and the first image of the next) and merges the two readings by box overlap (`merge_page_readings`, IoU >= 0.60, with a label-agreement fallback for regions where the two reads agree on a real label but disagree enough on the box to miss the IoU threshold — common on short, tightly packed regions like answer-key rows). One case remains unresolved: `mcq_answer_key` regions, which carry no label (always `null` per the gold-data convention), can still be read as two separate regions instead of merging into one.
- Continuation handling (`apply_continuation_chains`, for a region whose content spans a page boundary) is built and works on the one confirmed real example in the entire 80-paper dataset (`Anglo_Chinese-2519`, `oe_answer_key` label 12, page 9-10), but that's a single data point — not confirmed robust at full 80-paper scale.
- Not yet verified: missing cover-page detection, blank-page handling, multiple regions appearing in one screenshot/crop, and whether the `paper` attribute on a region is ever populated correctly (currently always `null` in observed output).

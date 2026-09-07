"""One-off lookup: prints every training run on this Tinker account, newest
first, with its most recent checkpoint (if any has been saved). Read-only --
just lists metadata, doesn't touch weights or spend training/sampling credits.

Run from teekathon\\main with: uv run python list_runs.py
"""

from dotenv import load_dotenv

load_dotenv()

import tinker

rest_client = tinker.ServiceClient().create_rest_client()
runs = rest_client.list_training_runs(limit=20).result().training_runs
runs.sort(key=lambda run: run.last_request_time, reverse=True)

if not runs:
    print("no training runs found on this account")

for run in runs:
    checkpoint = run.last_checkpoint
    checkpoint_info = (
        f"{checkpoint.tinker_path} (saved {checkpoint.time})"
        if checkpoint is not None
        else "no checkpoint saved yet"
    )
    print(
        f"run_id={run.training_run_id}  base_model={run.base_model}  "
        f"last_used={run.last_request_time}  checkpoint={checkpoint_info}"
    )

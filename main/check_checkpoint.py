"""Checks one run_id's checkpoints through list_checkpoints -- the same
dedicated per-run endpoint create_training_client's own resume logic uses,
as opposed to list_training_runs' bulk "last_checkpoint" summary field
(which list_runs.py relies on, and which may be stale or scoped
differently). Read-only.

Run from teekathon\\main with: uv run python check_checkpoint.py <run_id>
"""

import sys

from dotenv import load_dotenv

load_dotenv()

import tinker

if len(sys.argv) != 2:
    raise SystemExit("usage: uv run python check_checkpoint.py <run_id>")

run_id = sys.argv[1]
rest_client = tinker.ServiceClient().create_rest_client()

checkpoints = rest_client.list_checkpoints(run_id).result().checkpoints
if not checkpoints:
    print(f"{run_id}: 0 checkpoints found via list_checkpoints")
else:
    for checkpoint in checkpoints:
        print(
            f"{run_id}: {checkpoint.checkpoint_type} checkpoint  "
            f"id={checkpoint.checkpoint_id}  time={checkpoint.time}  "
            f"path={checkpoint.tinker_path}"
        )

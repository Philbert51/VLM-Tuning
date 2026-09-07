"""Total Tinker training spend across every run on this account within the
last 14 days (the max window one billing-usage request covers), broken down
per run_id. One query -- no need to pass individual run_ids in. Read-only,
lists billing metadata, doesn't spend anything.

Run from teekathon\\main with: uv run python total_usage.py
"""

from datetime import datetime, timedelta, timezone

import tinker

from main.main import TRAINING_DOLLARS_PER_MILLION_TOKENS

rest_client = tinker.ServiceClient().create_rest_client()

current_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
ending_before = current_hour + timedelta(hours=1)
starting_on = ending_before - timedelta(days=14)

usage = rest_client.get_billing_usage(starting_on, ending_before).result()

tokens_by_session: dict[str, int] = {}
base_models_seen: set[str] = set()
total_training_tokens = 0

for event in usage.data:
    if event.event_info.type != "training":
        continue
    base_models_seen.add(event.base_model)
    tokens_by_session[event.session_id] = (
        tokens_by_session.get(event.session_id, 0) + event.event_info.token_count
    )
    total_training_tokens += event.event_info.token_count

print("per run_id (session_id:train:0):")
for session_id, tokens in sorted(tokens_by_session.items(), key=lambda item: -item[1]):
    dollars = (tokens / 1_000_000) * TRAINING_DOLLARS_PER_MILLION_TOKENS
    print(f"  {session_id}:train:0  tokens={tokens}  ${dollars:.6f}")

total_dollars = (total_training_tokens / 1_000_000) * TRAINING_DOLLARS_PER_MILLION_TOKENS
print(f"total training tokens (last 14 days): {total_training_tokens}")
print(f"total training spend (last 14 days): ${total_dollars:.6f}")

if len(base_models_seen) > 1:
    print(f"note: multiple base models in this window ({base_models_seen}) -- "
          f"the ${TRAINING_DOLLARS_PER_MILLION_TOKENS}/M rate only applies to one of them")
if total_training_tokens == 0:
    print("billing data can lag by several hours or fall outside the 14-day window")

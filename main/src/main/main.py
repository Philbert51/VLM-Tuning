from pathlib import Path
import sys
from dataclasses import dataclass
from collections.abc import Sequence
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, field_validator
from typing import Literal, cast
from dotenv import load_dotenv

from transformers import PreTrainedTokenizer, AutoImageProcessor

from datetime import datetime, timedelta, timezone

import argparse
import pymupdf
import re
import tinker
import json
import os
import math
import cv2  
import inspect
import traceback

#initialization
# <modifiable>
folderPath = Path(__file__).resolve().parents[2] / "files" #desired folder path to scan the files, defaults to main/files
outputPath = Path(__file__).resolve().parents[2] / "output" #desired folder for output
max_output_tokens = 2000 #max tokens the VLM can generate per page, raise if responses get cut off
vlm_prompt_text = """You are given image(s) from a multi-page exam paper.

Output JSON only. You may output multiple objects.

Top-level fields:
- page (dtype: integer > 0): this page's position among the image(s) given in this call.

Region types:
- cover_page: cover/title content (title, subject, date, instructions, marks table). Box its actual visual extent, same as any other region. If nothing else except cover page is present on that specific page, mark the whole box as [0,0,1000,1000] and "label" : null. Judge this per page, not across every image given in this call.
- mcq_question: Multiple Choice Question, a multiple-choice question.
- oe_question: Open Ended, an open-ended question.
- mcq_answer_key: Multiple Choice Question Answer Key , an answer-key entry for a multiple-choice question. mcq_answer_key is the only region type that may be boxed in bulk, covering multiple questions' answers in a single box. When boxed this way, label must be null.
- oe_answer_key: Open Ended Answer Key, an answer-key entry for an open-ended question.


Each region also has:
- label (dtype: string | null) : the printed question number/label, or null if there isn't one.
- paper (dtype: int | null) : printed paper number. leave null if the page does not contain the information or unsure.
- continuation (dtype: enum["start","middle", "end", "single"]): "single" when the region fits on one page. When a question or answer spans multiple pages, output one box per page with the same type, paper, and label, and set "start", "middle" or "end" in page order.
- box_2d (dtype: list of 4 integers): the region's bounding box as [ymin, xmin, ymax, xmax], normalized to integers 0-1000. (0,0) is the page's top-left corner, y increases downward, x increases rightward.
- If uncertain about the exact boundary, extend into surrounding blank whitespace rather than cutting into the question's text.

If a page has none of the region types listed, still output an object for that page with an empty regions list, for example {"page": n, "regions": []} where n = page number. Do not invent a region.
Child questions (lettered sub-parts, e.g. 7(a), 7(b)) merge into one region under the parent question's number (e.g. "7"). Never one region per child question.

<output_example>
[
  {
    "page": 1,
    "regions": [
      {
        "type": "mcq_question",
        "paper": 1,
        "label": null,
        "continuation": "single",
        "box_2d": [500, 100, 800, 900]
      }
    ]
  },
  {
    "page": 2,
    "regions": [
      {
        "type": "mcq_question",
        "paper": 1,
        "label": "2",
        "continuation": "single",
        "box_2d": [100, 100, 400, 900]
      }
    ]
  }
]
</output_example>

"""
# </modifiable>
_image_processor = None
continuation = Literal["start","middle", "end", "single"]
samplingTemperature = 0.0 #0 = deterministic/repeatable output, higher = more varied output. this setting shouldn't be changed.
load_dotenv()
# defaultBaseModel : str = "Qwen/Qwen3.5-4B"
defaultBaseModel : str = "Qwen/Qwen3.6-35B-A3B"

# Current training price for the configured base model, per million tokens.
TRAINING_DOLLARS_PER_MILLION_TOKENS = 1.177

RegionType = Literal["cover_page", "mcq_question", "oe_question", "mcq_answer_key", "oe_answer_key"]

#initialization

def testing(argv: list[str] | None = None)  :
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--run-id", type=str, required=True)
    # args = parser.parse_args(argv)
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    checkpoints = rest_client.list_checkpoints().result().checkpoints
    print(checkpoints)


class _Tee:
    """Mirrors write()/flush() calls to every stream passed in -- used to
    send print() output to both the terminal and a log file at once."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main(argv: list[str] | None = None) -> int:
    args = build_main_parser().parse_args(argv)

    # create folder if it doesn't already exist, exist_ok make it so-
    # that the program doesn't crash if it already exists
    folderPath.mkdir(parents=True, exist_ok=True)

    # check if file exists
    fileName = args.image_path
    filePath = folderPath / args.image_path
    if not file_exists_in_folder(fileName) :
        raise FileNotFoundError(f"File {filePath} does not exist")

    # Mirror every print() into log.txt next to the PDF being processed
    log_path = filePath.parent / "log.txt"
    log_file = open(log_path, "a", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log_file)
    print(f"logging to {log_path}")

    # Write rendered pages, overlays and crops next to the PDF being
    # processed instead of the fixed global output folder keeps a run's
    # output self-contained alongside its source file, same as log.txt.
    run_output_dir = filePath.parent / "output"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # render the page
        rendered_pages = render_pdf(pdf_path=filePath, output_dir=run_output_dir)
        print(f"rendered {len(rendered_pages)} page(s)")

        # Index rendered pages by real PDF page number for final materialization.
        pages_by_number = {rendered_page.page_number: rendered_page for rendered_page in rendered_pages}

        # Preserve separate readings so duplicate interior-page readings can be merged.
        page_readings: dict[int, list[list[DetectedRegion]]] = {}
        if len(rendered_pages) == 1:
            # Preserve single-page sampling when no adjacent pair exists.
            rendered_page = rendered_pages[0]
            print(f"sampling page {rendered_page.page_number}")
            raw_text = sample_vlm(
                vlm_prompt_text,
                defaultBaseModel,
                image_paths=[rendered_page.path],
                run_id=args.run_id,
            )
            print(f"page {rendered_page.page_number}: raw_text={raw_text!r}")
            page_detections = extract_page_detections(cast(str, raw_text))
            print(f"page {rendered_page.page_number}: found {len(page_detections.regions)} region(s)")
            page_readings[rendered_page.page_number] = [page_detections.regions]
        else:
            for page_a, page_b in consecutive_page_pairs(rendered_pages):
                print(f"sampling pages {page_a.page_number}-{page_b.page_number}")
                # Send one overlapping adjacent-page window per model call.
                raw_text = sample_vlm(
                    vlm_prompt_text,
                    defaultBaseModel,
                    image_paths=[page_a.path, page_b.path],
                    run_id=args.run_id,
                )
                print(f"pages {page_a.page_number}-{page_b.page_number}: raw_text={raw_text!r}")
                # Parse every complete page object returned for the pair.
                pair_detections = extract_page_detections_batch(cast(str, raw_text))
                print(f"pages {page_a.page_number}-{page_b.page_number}: received {len(pair_detections)} page object(s)")

                # Reject a response containing no usable page object.
                if not pair_detections:
                    raise ValueError(
                        f"expected at least 1 page object for real page pair "
                        f"({page_a.page_number}, {page_b.page_number}), received 0"
                    )

                # Reject responses exceeding the two supplied pages.
                if len(pair_detections) > 2:
                    raise ValueError(
                        f"expected at most 2 page objects for real page pair "
                        f"({page_a.page_number}, {page_b.page_number}), "
                        f"received {len(pair_detections)}"
                    )

                # Batch-relative pages 1 and 2 identify the corresponding input image.
                relative_pages = [detections.page for detections in pair_detections]
                if (
                    any(relative_page not in {1, 2} for relative_page in relative_pages)
                    or len(set(relative_pages)) != len(relative_pages)
                ):
                    raise ValueError(
                        f"expected unique batch-relative pages 1 or 2 for real page pair "
                        f"({page_a.page_number}, {page_b.page_number}), "
                        f"received {relative_pages}"
                    )
                detections_by_relative_page = {
                    detections.page: detections for detections in pair_detections
                }
                # Convert batch-relative pages back to real PDF page numbers.
                for relative_page, rendered_page in ((1, page_a), (2, page_b)):
                    detections = detections_by_relative_page.get(relative_page)
                    # A missing side represents an empty reading for that page.
                    regions = [] if detections is None else detections.regions
                    page_readings.setdefault(rendered_page.page_number, []).append(regions)

        print(f"merging readings for {len(page_readings)} page(s)")
        # Merge repeated readings before materializing each real PDF page.
        all_detections = [
            PageDetections(
                page=page_number,
                regions=merge_page_readings(page_readings[page_number]),
            )
            for page_number in sorted(page_readings)
        ]

        print("applying continuation chains...")
        # Fix middle/end labels along real page order using each chain's start label.
        all_detections = apply_continuation_chains(all_detections)

        # Same all_detections list the materialize loop below reads from,
        # written out as-is so predictions can be inspected/diffed without
        # opening the overlay images.
        predictions_path = run_output_dir / "predictions.json"
        predictions_path.write_text(
            json.dumps(
                [detections.model_dump(mode="json") for detections in all_detections],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote predictions to {predictions_path}")

        # Score this run against ground truth when the source PDF has a sibling gold.json 
        gold_path = filePath.parent / "gold.json"
        if gold_path.exists():
            try:
                evaluation_report = evaluate_predictions_against_gold(gold_path, all_detections)
                if evaluation_report:
                    evaluation_path = run_output_dir / "evaluation.json"
                    evaluation_path.write_text(
                        json.dumps(evaluation_report, indent=2),
                        encoding="utf-8",
                    )
                    print(f"wrote evaluation to {evaluation_path}")
            except Exception:
                print(f"CRASH evaluating against {gold_path}:", flush=True)
                print(traceback.format_exc(), flush=True)

        # draw overlays and save cropped regions for each page's detections
        for detections in all_detections:

            # Reject a detection without a real PDF page number before lookup.
            if detections.page is None:
                raise ValueError("page detection is missing a real PDF page number")

            rendered_page = pages_by_number.get(detections.page)
            if rendered_page is None:
                print(f"no rendered page found for page_number={detections.page}")
            else :
                print(f"materializing page {detections.page}: {len(detections.regions)} region(s)")
                materialize_page_detections(rendered_page.path, detections, run_output_dir)

        return 0
    except Exception:
        print(f"CRASH processing {filePath}:", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        sys.stdout = original_stdout
        log_file.close()

def train(argv: list[str] | None = None) :
    args = build_train_parser().parse_args(argv)
    print(args.training_folder_path)

    # Mirror every print() into log.txt inside the training folder 
    log_path = Path(args.training_folder_path) / "log.txt"
    log_file = open(log_path, "a", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log_file)
    print(f"logging to {log_path}")

    try:
        # gold.json/gold/ crops -> TrainingExample objects, one per annotated
        # page. render_path holds the rendered PNGs used to template-match each
        # gold crop back to its page.
        training_examples = build_training_examples_from_gold(
            data_path=Path(args.training_folder_path),
            render_path=outputPath / "renders",
        )

        # --limit slices down to the first N examples pass --limit 1 to
        # smoke-test against a single page before committing to everything
        # build_training_examples_from_gold found.
        if args.limit is not None:
            training_examples = training_examples[: args.limit]
        print(f"training on {len(training_examples)} example(s)")

        # Create a new run or resume the requested training run.
        training_client = create_training_client(run_id=args.run_id)

        print(f"training client ready, run_id={training_client.model_id}")

        # Full cost estimate (text + image tokens)
        tokenizer = training_client.get_tokenizer()
        text_token_count = sum(
            len(tokenizer.encode(example.inputText))
            + len(tokenizer.encode(example.expectedOutput))
            for example in training_examples
        )


        image_processor = get_image_processor()
        image_token_count_by_path: dict[Path, int] = {}
        image_token_count = 0
        for example in training_examples:
            for image_path in example.image_paths or []:
                if image_path not in image_token_count_by_path:
                    image_token_count_by_path[image_path] = compute_expected_tokens(
                        image_processor, image_path
                    )
                image_token_count += image_token_count_by_path[image_path]

        # train_vlm runs this same full batch once per step (see --num-steps), so
        # the per-pass token count above must be scaled by step count otherwise
        # this estimate silently undercounts total spend by a factor of num_steps.
        total_token_count = text_token_count + image_token_count
        estimated_total_cost = (
            total_token_count * args.num_steps / 1_000_000
        ) * TRAINING_DOLLARS_PER_MILLION_TOKENS
        print(
            f"tokens per step: {text_token_count} text + {image_token_count} image "
            f"= {total_token_count}, over {args.num_steps} step(s), "
            f"estimated cost: ${estimated_total_cost:.4f}"
        )

        # Explicit go/no-go, now with a real cost figure already in hand
        # a mistyped argument or an unexpectedly high estimate both get
        # caught here, before train_vlm actually starts spending on
        # forward_backward/optim_step calls.
        print("confirm training configuration:")
        print(f"  training_folder_path: {args.training_folder_path}")
        print(f"  resume run_id: {args.run_id}")
        print(f"  limit: {args.limit}")
        print(f"  num_steps: {args.num_steps}")
        print(f"  learning_rate: {args.learning_rate}")
        print(f"  loss_fn: cross_entropy")
        print(f"  save_sampler_weights: {args.save_sampler_weights}")
        confirmation = input("proceed with this configuration? [y/N]: ").strip().lower()
        if confirmation != "y":
            print("training cancelled")
            return

        print(
            f"starting training: {len(training_examples)} example(s), "
            f"{args.num_steps} step(s), learning_rate={args.learning_rate}"
        )
        train_vlm(
            training_examples,
            training_client,
            num_steps=args.num_steps,
            learning_rate=args.learning_rate,
        )
        print("training loop finished, saving checkpoint...")

        # persist the fine-tuned weights so a later `main --run-id=...` call can
        # find them through create_sampling_client's list_checkpoints lookup.
        save_result = training_client.save_state("checkpoint-final", overwrite=True).result()
        print(f"run_id: {training_client.model_id}")
        print(f"checkpoint saved to: {save_result.path}")

        if args.save_sampler_weights:
            sampler_result = training_client.save_weights_for_sampler("checkpoint-final").result()
            print(f"sampler weights saved to: {sampler_result.path}")
    except Exception:
        print(
            f"CRASH during training (training_folder_path={args.training_folder_path}, "
            f"resume run_id={args.run_id}):",
            flush=True,
        )
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        sys.stdout = original_stdout
        log_file.close()


def spendings(argv: list[str] | None = None) -> int:
    args = build_spendings_parser().parse_args(argv)
    if not os.environ.get("TINKER_API_KEY"):
        raise ValueError("TINKER_API_KEY is missing. Add it to .env or export it.")

    # A training model ID uses the session prefix before ':train:'.
    session_id = args.run_id.split(":train:", 1)[0]

    # Query the largest billing window accepted by one SDK request.
    current_hour = datetime.now(timezone.utc).replace(
        minute=0,
        second=0,
        microsecond=0,
    )
    ending_before = current_hour + timedelta(hours=1)
    starting_on = ending_before - timedelta(days=14)

    # Fetch organization billing events through the authenticated SDK client.
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    usage = rest_client.get_billing_usage(
        starting_on,
        ending_before,
    ).result()

    # Keep training-token events belonging to the requested session only.
    training_tokens = 0
    for event in usage.data:
        if event.session_id != session_id:
            continue
        if event.event_info.type != "training":
            continue
        if event.base_model != defaultBaseModel:
            raise ValueError(
                f"no configured training price for base_model={event.base_model!r}"
            )
        training_tokens += event.event_info.token_count

    # Convert billed training tokens into a dollar estimate.
    usage_dollars = (
        training_tokens / 1_000_000
    ) * TRAINING_DOLLARS_PER_MILLION_TOKENS
    print(f"training tokens: {training_tokens}")
    print(f"usage dollars: ${usage_dollars:.6f}")
    if training_tokens == 0:
        print("billing data can lag by several hours or fall outside the 14-day window")
    return 0

#region classes

#from detector.py
@dataclass(frozen=True)
class RenderedPage:
    page_number: int
    path: Path
    width: int
    height: int

#from detector.py
class DetectedRegion(BaseModel):
    type : RegionType = Field(description="The kind of question or answer region.")
    paper : int | None = Field(default = None, description="nullable int field for paper")
    label : str | None = Field(
        default=None,
        max_length=32,
        description="Printed question number, or null for bulk MCQ answer table.",
    )
    continuation : Literal["start", "middle", "end", "single"] = Field(default="single", description="type of continuation")
    box_2d: Sequence[int] = Field(
        min_length=4,
        max_length=4,
        description=(
            "Bounding box [ymin, xmin, ymax, xmax], normalized to integers 0-1000."
        ),
    )

    @field_validator("box_2d")
    @classmethod
    def validate_box(cls, box: list[int]) -> list[int]:
        ymin, xmin, ymax, xmax = box
        if any(coordinate < 0 or coordinate > 1000 for coordinate in box):
            raise ValueError("box coordinates must be between 0 and 1000")
        if ymin >= ymax or xmin >= xmax:
            raise ValueError("box must have positive width and height")
        return box
    
class PageDetections(BaseModel):
        # ge = greater than or equal to 
    page : int | None = Field(default = None)
    regions: list[DetectedRegion] = Field(default_factory=list, max_length=100)

@dataclass(frozen=True)
class TrainingExample:
    """One labeled example for train_vlm: an input paired with its correct output.

    Args:
        inputText: str
        expectedOutput: str
        image_paths: list[Path]
    """

    inputText: str
    expectedOutput: str
    image_paths: list[Path] | None = None


#endregion classes

#region functions

def build_main_parser() -> argparse.ArgumentParser:
    # Defines the "--message" flag required, must be a string.
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    return parser

def build_train_parser() -> argparse.ArgumentParser:
    # Defines the "--message" flag required, must be a string.
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-folder-path", type=str, required=True)

    # caps how many TrainingExample objects actually go into train_vlm's
    parser.add_argument("--limit", type=int, default=None)

    # How many times train_vlm repeats forward_backward/optim_step over the whole batch.
    #  no mini-batching or epochs yet, so this is the only knob
    # controlling how much the model actually learns versus just proving the
    # plumbing works.
    parser.add_argument("--num-steps", type=int, default=20)

    # optim_step's step size. LoRA needs a much bigger learning rate than full fine-tuning
    parser.add_argument("--learning-rate", type=float, default=0.0005)

    # A missing run ID starts new training; a supplied run ID resumes training.
    parser.add_argument("--run-id", type=str, default=None)

    # save_state alone only produces a training checkpoint, not something
    # main() can sample from pass this to also call
    # save_weights_for_sampler right after, in the same run.
    parser.add_argument("--save-sampler-weights", action="store_true")

    return parser


def build_spendings_parser() -> argparse.ArgumentParser:
    # Require one training run so unrelated organization usage stays excluded.
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    return parser


# from render.py
def render_pdf(pdf_path: Path, output_dir : Path, dpi: int = 200, page_limit: int | None = None) -> list[RenderedPage]:
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if page_limit is not None and page_limit <= 0:
        raise ValueError("page_limit must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[RenderedPage] = []
    with pymupdf.open(pdf_path) as document:
        page_count = document.page_count
        if page_limit is not None:
            page_count = min(page_count, page_limit)

        for page_index in range(page_count):
            page = document[page_index]
            pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
            path = output_dir / f"page-{page_index + 1}.png"
            pixmap.save(path)
            renderedPage = RenderedPage(
                page_number=page_index + 1,
                path=path,
                width=pixmap.width,
                height=pixmap.height,
            )
            verify_page_dpi(renderedPage, dpi = dpi)
            rendered.append(
                renderedPage
            )
    return rendered


def build_prompt(tokenizer: PreTrainedTokenizer,text: str,image_paths: list[Path] | None = None,image_format: Literal["png", "jpeg"] = "png",
) -> tinker.types.ModelInput:
    prompt = tinker.types.ModelInput.from_ints(tokenizer.encode(text))

    image_processor = get_image_processor()
    for index, image_path in enumerate(image_paths or [], start=1):
        # Nothing else marks which image is which position the prompt's
        # own "page" field means position among these images, so each one
        # needs a label the model can actually see, not just its raw order.
        label_tokens = tokenizer.encode(f"\nImage {index}:")
        prompt = prompt.append(tinker.types.EncodedTextChunk(tokens=label_tokens))

        expected_tokens = compute_expected_tokens(image_processor, image_path)
        prompt = prompt.append(
            tinker.types.ImageChunk(
                data=image_path.read_bytes(),
                format=image_format,
                expected_tokens=expected_tokens,
            )
        )

    return prompt

def calculate_box_entropy_loss(
    gold_box: Sequence[int],
    model_box: Sequence[int],
) -> float:
    iou = box_iou(gold_box, model_box)
    if iou >= 1.0:
        return 0.0
    return -math.log(max(iou, 1e-7))


def create_training_client(
    base_model: str = defaultBaseModel,
    rank: int = 32,
    run_id: str | None = None,
) -> tinker.TrainingClient:
    #base_model = the parameter
    #if key not found
    if not os.environ.get("TINKER_API_KEY"):
        raise ValueError("TINKER_API_KEY is missing. Add it to .env or export it.")

    #initialize the client, already automatically grabs TINKER_API_KEY
    service_client = tinker.ServiceClient()

    # Start a fresh LoRA run when no previous run was requested.
    if run_id is None:
        return service_client.create_lora_training_client(
            base_model=base_model,
            rank=rank,
        )

    # Load every checkpoint associated with the requested run.
    rest_client = service_client.create_rest_client()
    checkpoints = rest_client.list_checkpoints(run_id).result().checkpoints

    # Optimizer resumption requires a training checkpoint, not sampler weights.
    training_checkpoints = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.checkpoint_type == "training"
    ]
    if not training_checkpoints:
        raise ValueError(f"no training checkpoints found for run_id={run_id!r}")

    # Resume from the newest available training state.
    latest_checkpoint = max(
        training_checkpoints,
        key=lambda checkpoint: checkpoint.time,
    )
    print(
        f"resuming run_id={run_id} from checkpoint="
        f"{latest_checkpoint.tinker_path}"
    )
    return service_client.create_training_client_from_state_with_optimizer(
        latest_checkpoint.tinker_path
    )


def sample_runid(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Save sampler weights for an existing training run's latest checkpoint."
    )
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--name", type=str, default="checkpoint-final")
    args = parser.parse_args(argv)

    # Reuses train()'s own resume path loads the requested run's latest
    # training checkpoint (weights + optimizer) into a new client, without
    # running forward_backward/optim_step again.
    training_client = create_training_client(run_id=args.run_id)
    print(f"loaded training checkpoint for run_id={args.run_id}")

    sampler_result = training_client.save_weights_for_sampler(args.name).result()
    print(f"new run_id: {training_client.model_id}")
    print(f"sampler weights saved to: {sampler_result.path}")
    return 0


def training_run(argv: list[str] | None = None) -> int:
    # Read the run ID supplied through the command line.
    args = build_training_run_parser().parse_args(argv)
    run = get_training_run(args.run_id)
    # Print all returned run details in readable JSON.
    print(run.model_dump_json(indent=2))
    return 0


def build_training_run_parser() -> argparse.ArgumentParser:
    # Define arguments for training-run lookup.
    parser = argparse.ArgumentParser(description="Show Tinker training-run details.")
    parser.add_argument("--run-id", type=str, required=True)
    return parser


def get_training_run(run_id: str) -> tinker.types.TrainingRun:
    # Require configured credentials before contacting Tinker.
    if not os.environ.get("TINKER_API_KEY"):
        raise ValueError("TINKER_API_KEY is missing. Add it to .env or export it.")

    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    # Wait for the requested run details and return the response.
    return rest_client.get_training_run(run_id).result()



def create_sampling_client(base_model: str, run_id: str | None = None) -> tinker.SamplingClient:
    if not os.environ.get("TINKER_API_KEY"):
        raise ValueError("TINKER_API_KEY is missing. Add it to .env or export it.")

    service_client = tinker.ServiceClient()

    if run_id is None:
        print(f"sampling client: using base model {base_model}, no checkpoint")
        return service_client.create_sampling_client(base_model=base_model)

    # No weights are loadable from run_id alone until save_state/save_weights_for_sampler actually ran for it.
    rest_client = service_client.create_rest_client()
    checkpoints = rest_client.list_checkpoints(run_id).result().checkpoints
    if not checkpoints:
        raise ValueError(f"no checkpoints found for run_id={run_id!r}")

    # Picks the most recently saved checkpoint for this run.
    latest_checkpoint = max(checkpoints, key=lambda checkpoint: checkpoint.time)
    print(f"sampling client: using checkpoint {latest_checkpoint.tinker_path}")
    return service_client.create_sampling_client(model_path=latest_checkpoint.tinker_path)


def sample_vlm(
    text: str,
    base_model : str,
    image_paths: list[Path] | None = None,
    run_id: str | None = None,
    num_samples: int = 1,
) -> str | Sequence[str]:
    sampling_client = create_sampling_client(base_model, run_id)

    tokenizer = sampling_client.get_tokenizer()
    prompt = build_prompt(tokenizer, text, image_paths)
    sampling_params = tinker.types.SamplingParams(max_tokens=max_output_tokens, temperature=samplingTemperature)

    print(f"sample_vlm: sending request ({len(image_paths) if image_paths else 0} image(s))")
    # temperature stays 0 only one correct output exists among the candidates.
    # sample() sends the request and returns right away; .result() blocks until the server responds with response.
    response = sampling_client.sample(prompt=prompt, num_samples=num_samples, sampling_params=sampling_params).result()
    print("sample_vlm: response received")

    decoded_samples = [cast(str, tokenizer.decode(sequence.tokens)) for sequence in response.sequences]
    return decoded_samples[0] if num_samples == 1 else decoded_samples


def train_vlm(
    training_examples: list[TrainingExample],
    training_client: tinker.TrainingClient,
    *,
    loss_fn: tinker.types.LossFnType = "cross_entropy",
    learning_rate: float = 0.0005,
    num_steps: int = 1,
) -> list[tinker.types.ForwardBackwardOutput]:
    tokenizer = training_client.get_tokenizer()
    print(f"building {len(training_examples)} training datum(s)...")

    training_data = []
    for example in training_examples:
        prompt_input = build_prompt(tokenizer, example.inputText, example.image_paths)
        answer_tokens = tokenizer.encode(example.expectedOutput)

        # Nothing else in this sequence ever tells the model to stop
        # training the eos token right after the answer is what gives
        # sample_vlm's implicit EOS stopping (SamplingParams.stop=None)
        # something to actually stop on.
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has no eos_token_id -- can't train a stop signal")
        answer_tokens = answer_tokens + [tokenizer.eos_token_id]

        # cross_entropy predicts each position against the token right after
        # it, across the whole sequence model_input and target_tokens have
        # to be the same length. Build one sequence covering prompt then
        # completion, weight the prompt at 0 (nothing learned there) and the
        # completion at 1.0, then shift by one: model_input drops the last
        # token, target_tokens drops the first. An image chunk's positions
        # count as target token 0 never trained, weight already 0 there.
        full_chunks = list(prompt_input.chunks) + [
            tinker.types.EncodedTextChunk(tokens=answer_tokens)
        ]
        full_weights = [0.0] * prompt_input.length + [1.0] * len(answer_tokens)

        last_chunk = full_chunks[-1]
        input_chunks = full_chunks[:-1]
        if last_chunk.length > 1:
            input_chunks.append(tinker.types.EncodedTextChunk(tokens=last_chunk.tokens[:-1]))

        all_tokens: list[int] = []
        for chunk in full_chunks:
            if isinstance(chunk, tinker.types.EncodedTextChunk):
                all_tokens.extend(chunk.tokens)
            else:
                all_tokens.extend([0] * chunk.length)
        target_tokens = all_tokens[1:]
        weights = full_weights[1:]

        training_data.append(
            tinker.types.Datum(
                model_input=tinker.types.ModelInput(chunks=input_chunks),
                loss_fn_inputs=cast(
                    tinker.types.LossFnInputs,
                    {
                        "target_tokens": tinker.types.TensorData(
                            data=target_tokens, dtype="int64", shape=[len(target_tokens)]
                        ),
                        "weights": tinker.types.TensorData(
                            data=weights, dtype="float32", shape=[len(weights)]
                        ),
                    },
                ),
            )
        )

    print(f"built {len(training_data)} datum(s), starting {num_steps} step(s)")
    results: list[tinker.types.ForwardBackwardOutput] = []
    # Runs training_data whole, num_steps times no batching or multi-epoch handling yet.
    for step in range(num_steps):
        print(f"step {step + 1}/{num_steps}: submitting forward_backward")
        # "cross_entropy" (the default) is the -(log(p)) per-token loss; this call also computes the gradients.
        forward_backward_future = training_client.forward_backward(training_data, loss_fn)

        print(f"step {step + 1}/{num_steps}: submitting optim_step")
        # Applies forward_backward's gradients through Adam; optim_step itself takes no data argument.
        optim_step_future = training_client.optim_step(tinker.types.AdamParams(learning_rate=learning_rate))

        # Both futures are created before either blocks; the server orders them via seq_id, not call order here.
        print(f"step {step + 1}/{num_steps}: waiting on forward_backward result")
        forward_backward_result = forward_backward_future.result()
        print(f"step {step + 1}/{num_steps}: waiting on optim_step result")
        optim_step_future.result()

        results.append(forward_backward_result)

        # cross_entropy's per-datum output carries "logprobs" log(p) for
        # the correct token at each position, matching that datum's own
        # target_tokens. Negating and weighting by the same per-token
        # weights sent in loss_fn_inputs (0 for the prompt, 1 for the
        # completion) turns this back into the loss cross_entropy actually
        # trains against, averaged over the whole step's batch. Wrapped so
        # a metrics-only bug here can never abort a real training run
        # forward_backward and optim_step have both already completed by
        # this point regardless of what happens below.
        try:
            total_weighted_nll = 0.0
            total_weight = 0.0
            for datum, loss_fn_output in zip(training_data, forward_backward_result.loss_fn_outputs):
                logprobs = loss_fn_output["logprobs"].data
                weights = datum.loss_fn_inputs["weights"].data
                for logprob, weight in zip(logprobs, weights):
                    total_weighted_nll += -logprob * weight
                    total_weight += weight
            step_loss = total_weighted_nll / total_weight if total_weight else float("nan")
            print(f"step {step + 1}/{num_steps} done, loss={step_loss:.4f}")
        except Exception:
            print(f"step {step + 1}/{num_steps} done, loss unavailable:", flush=True)
            print(traceback.format_exc(), flush=True)

    return results

# PageDetections changed doesn't effect because this only extracts
def extract_page_detections(raw_text : str) -> PageDetections:
    # manual brace-depth scan, finds the first complete top-level {...}
    # object in raw_text. Single page version.
    depth = 0
    start = None
    in_string = False
    escape_next = False

    for index, char in enumerate(raw_text):
        if escape_next:
            escape_next = False
            continue

        if char == "\\" and in_string:
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string # boolean switch
            continue

        if in_string:
            continue

        if char == "{":
            if depth == 0:
                start = index
            depth += 1

        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                object_text = raw_text[start:index + 1]
                return PageDetections.model_validate_json(object_text)

    print(f"extract_page_detections: no complete JSON object found, raw_text={raw_text!r}")
    raise ValueError("No complete JSON object found in VLM response")


def extract_page_detections_batch(raw_text: str) -> list[PageDetections]:
    # manual brace-depth scan, finds every complete top-level {...} object
    # in raw_text. multiple objects version returns a list instead
    detections = []
    depth = 0
    start = None
    in_string = False
    escape_next = False

    for index, char in enumerate(raw_text):
        if escape_next:
            escape_next = False
            continue

        if char == "\\" and in_string:
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            if depth == 0:
                start = index
            depth += 1

        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                object_text = raw_text[start:index + 1]
                detections.append(PageDetections.model_validate_json(object_text))
                start = None

    if not detections:
        print(f"extract_page_detections_batch: no complete JSON object found, raw_text={raw_text!r}")
    return detections


#from images.py
def normalized_box_to_pixels(box_2d: Sequence[int], *, width: int, height: int, padding: int,) -> tuple[int, int, int, int]:
    ymin, xmin, ymax, xmax = box_2d
    left = max(0, math.floor(xmin * width / 1000) - padding)
    top = max(0, math.floor(ymin * height / 1000) - padding)
    right = min(width, math.ceil(xmax * width / 1000) + padding)
    bottom = min(height, math.ceil(ymax * height / 1000) + padding)
    return left, top, right, bottom

# verified works
def box_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    # Intersection over Union between two [ymin, xmin, ymax, xmax] boxes.
    if len(box_a) != 4 or len(box_b) != 4:
        print(f"box_iou error: box_a and box_b must each have exactly 4 items, got {len(box_a)} and {len(box_b)}")
        return 0.0

    a_ymin, a_xmin, a_ymax, a_xmax = box_a
    b_ymin, b_xmin, b_ymax, b_xmax = box_b

    overlap_ymin = max(a_ymin, b_ymin)
    overlap_xmin = max(a_xmin, b_xmin)
    overlap_ymax = min(a_ymax, b_ymax)
    overlap_xmax = min(a_xmax, b_xmax)

    overlap_height = max(0, overlap_ymax - overlap_ymin)
    overlap_width = max(0, overlap_xmax - overlap_xmin)
    overlap_area = overlap_height * overlap_width

    box_a_area = (a_ymax - a_ymin) * (a_xmax - a_xmin)
    box_b_area = (b_ymax - b_ymin) * (b_xmax - b_xmin)
    union_area = box_a_area + box_b_area - overlap_area

    if union_area == 0:
        return 0.0

    return overlap_area / union_area




# Ground truth against which a run's predictions get scored, when the
# source PDF has a gold.json alongside it separate from
# build_training_examples_from_gold's gold reading, since this compares
# an existing PageDetections prediction list against gold instead of
# building TrainingExample objects.
def evaluate_predictions_against_gold(
    gold_path: Path,
    all_detections: Sequence[PageDetections],
    *,
    iou_threshold: float = 0.90,
) -> dict:
    with gold_path.open("r", encoding="utf-8") as gold_file:
        gold_data = json.load(gold_file)

    # Only corrected_data's list shape carries a ready box_2d directly
    # the older data/ dict shape stores fragment crop paths instead and
    # needs template-matching against rendered pages first, a heavier step
    # this comparison does not perform.
    if not isinstance(gold_data, list):
        print(
            f"evaluate_predictions_against_gold: {gold_path} is not "
            f"corrected_data's list format, skipping"
        )
        return {}

    gold_regions_by_page: dict[int, list[DetectedRegion]] = {}
    for entry in gold_data:
        page_number = entry["page"]
        for source_region in entry.get("regions", []):
            gold_regions_by_page.setdefault(page_number, []).append(
                DetectedRegion(
                    paper=source_region.get("paper"),
                    type=cast(RegionType, source_region["type"]),
                    label=source_region.get("label"),
                    continuation=source_region.get("continuation") or "single",
                    box_2d=source_region["box_2d"],
                )
            )

    predicted_regions_by_page = {
        detections.page: list(detections.regions) for detections in all_detections
    }

    total_gold = 0
    correct = 0
    # Every claimed pair gets logged here, correct or not, so evaluation.json
    # shows the iou behind a pass as well as a fail not only failures.
    correct_matches: list[dict] = []
    wrong_field: list[dict] = []
    missing: list[dict] = []
    extra: list[dict] = []

    all_pages = sorted(set(gold_regions_by_page) | set(predicted_regions_by_page))
    for page_number in all_pages:
        gold_regions = list(gold_regions_by_page.get(page_number, []))
        predicted_regions = list(predicted_regions_by_page.get(page_number, []))
        total_gold += len(gold_regions)

        # Greedy best-IoU-first pairing within matching type, so a gold
        # region claims whichever same-type prediction actually overlaps
        # it, not just the first same-type prediction found on the page.
        candidate_pairs = []
        for gold_index, gold_region in enumerate(gold_regions):
            for predicted_index, predicted_region in enumerate(predicted_regions):
                if predicted_region.type != gold_region.type:
                    continue
                iou = box_iou(gold_region.box_2d, predicted_region.box_2d)
                candidate_pairs.append((iou, gold_index, predicted_index))
        candidate_pairs.sort(key=lambda pair: pair[0], reverse=True)

        claimed_gold: set[int] = set()
        claimed_predicted: set[int] = set()
        for iou, gold_index, predicted_index in candidate_pairs:
            if gold_index in claimed_gold or predicted_index in claimed_predicted:
                continue
            claimed_gold.add(gold_index)
            claimed_predicted.add(predicted_index)

            gold_region = gold_regions[gold_index]
            predicted_region = predicted_regions[predicted_index]
            match_entry = {
                "page": page_number,
                "type": gold_region.type,
                "gold_label": gold_region.label,
                "predicted_label": predicted_region.label,
                "gold_paper": gold_region.paper,
                "predicted_paper": predicted_region.paper,
                "iou": iou,
            }
            if (
                predicted_region.label == gold_region.label
                and predicted_region.paper == gold_region.paper
                and iou >= iou_threshold
            ):
                correct += 1
                correct_matches.append(match_entry)
            else:
                wrong_field.append(match_entry)

        for gold_index, gold_region in enumerate(gold_regions):
            if gold_index not in claimed_gold:
                missing.append(
                    {"page": page_number, "type": gold_region.type, "label": gold_region.label}
                )
        for predicted_index, predicted_region in enumerate(predicted_regions):
            if predicted_index not in claimed_predicted:
                extra.append(
                    {"page": page_number, "type": predicted_region.type, "label": predicted_region.label}
                )

    report = {
        "iou_threshold": iou_threshold,
        "total_gold_regions": total_gold,
        "correct": correct,
        "correct_matches": correct_matches,
        "wrong_field": wrong_field,
        "missing": missing,
        "extra": extra,
        "score": correct / total_gold if total_gold else None,
    }
    print(
        f"evaluate_predictions_against_gold: {correct}/{total_gold} region(s) correct "
        f"(type+paper+label match, IoU>={iou_threshold}), "
        f"{len(wrong_field)} wrong, {len(missing)} missing, {len(extra)} extra"
    )
    return report


# Build overlapping adjacent-page windows in real page order.
def consecutive_page_pairs(
    rendered_pages: Sequence[RenderedPage],
) -> list[tuple[RenderedPage, RenderedPage]]:
    # Sort once so input order cannot change pair membership.
    sorted_pages = sorted(rendered_pages, key=lambda page: page.page_number)
    # Pair every page with the immediate following page.
    return [
        (sorted_pages[index], sorted_pages[index + 1])
        for index in range(len(sorted_pages) - 1)
    ]


# Merge the two independent readings produced for an interior page.
def merge_page_readings(
    readings: Sequence[Sequence[DetectedRegion]],
) -> list[DetectedRegion]:
    # Return no regions when no reading exists.
    if not readings:
        return []
    # Skip duplicate matching when only one reading exists.
    if len(readings) == 1:
        return list(readings[0])
    if len(readings) != 2:
        raise ValueError(f"expected at most 2 readings for one page, received {len(readings)}")

    reading_a = readings[0]
    reading_b = readings[1]
    print(f"merge_page_readings: reading_a has {len(reading_a)}, reading_b has {len(reading_b)}")
    claimed_b_indexes: set[int] = set()
    merged_regions: list[DetectedRegion] = []

    # Match each first reading region against one unclaimed second reading region.
    for region_a in reading_a:
        matched_b_index: int | None = None
        for b_index, region_b in enumerate(reading_b):
            if b_index in claimed_b_indexes:
                continue
            if region_a.type != region_b.type:
                continue
            iou = box_iou(region_a.box_2d, region_b.box_2d)
            if iou < 0.60 and not (
                (region_a.label is not None and region_a.label == region_b.label)
                or (
                    region_a.type == "mcq_answer_key"
                    and region_a.label is None
                    and region_b.label is None
                    and iou >= 0.350
                )
            ):
                continue
            matched_b_index = b_index
            break

        if matched_b_index is None:
            # Preserve a region detected by only the first reading.
            merged_regions.append(region_a)
            continue

        claimed_b_indexes.add(matched_b_index)
        region_b = reading_b[matched_b_index]
        # Prefer continuation evidence over a conflicting single-page label.
        if region_a.continuation == "single" and region_b.continuation != "single":
            merged_regions.append(region_b)
        else:
            merged_regions.append(region_a)

    # Preserve a region detected by only the second reading.
    for b_index, region_b in enumerate(reading_b):
        if b_index not in claimed_b_indexes:
            merged_regions.append(region_b)

    print(f"merge_page_readings: {len(claimed_b_indexes)} matched, {len(merged_regions)} region(s) total")
    return merged_regions

# Corrects a middle/end region's label using its chain's start-page label.
def apply_continuation_chains(
    all_detections: Sequence[PageDetections],
) -> list[PageDetections]:
    # Keyed by (type, paper); each open entry holds the label captured when
    # that chain's "start" region was seen. Assumes at most one open chain
    # per (type, paper) at a time unconfirmed at full 80-paper scale.
    open_chains: dict[tuple[RegionType, int | None], str | None] = {}
    corrected_detections: list[PageDetections] = []

    # all_detections already comes in real page order from main()'s merge step.
    for detections in all_detections:
        corrected_regions: list[DetectedRegion] = []
        for region in sorted(detections.regions, key=lambda region: region.box_2d[0]):
            chain_key = (region.type, region.paper)

            if region.continuation == "start":
                # Anchor the chain on this page's own printed, trustworthy label.
                open_chains[chain_key] = region.label
                corrected_regions.append(region)
                continue

            if region.continuation in ("middle", "end"):
                anchor_label = open_chains.get(chain_key)
                if anchor_label is not None:
                    region = region.model_copy(update={"label": anchor_label})
                if region.continuation == "end":
                    # Chain finished; drop it so a later, unrelated region
                    # of the same (type, paper) doesn't inherit this label.
                    open_chains.pop(chain_key, None)

            corrected_regions.append(region)

        print(f"apply_continuation_chains: page {detections.page}: {len(corrected_regions)} region(s)")
        corrected_detections.append(
            PageDetections(page=detections.page, regions=corrected_regions)
        )

    return corrected_detections


# Construct the same adjacent-page label shape for both gold-data formats.
def build_page_pair_training_examples(
    *,
    input_text: str,
    regions_by_page: dict[int, list[DetectedRegion]],
    pages_by_number: dict[int, RenderedPage],
) -> list[TrainingExample]:
    # Use real page order before assigning batch-relative page numbers.
    rendered_pages = sorted(
        pages_by_number.values(),
        key=lambda page: page.page_number,
    )
    if not rendered_pages:
        return []
    if len(rendered_pages) == 1:
        # Preserve the single-image training shape for a one-page paper.
        rendered_page = rendered_pages[0]
        page_detections = PageDetections(
            page=1,
            regions=regions_by_page.get(rendered_page.page_number, []),
        )
        return [
            TrainingExample(
                inputText=input_text,
                expectedOutput=page_detections.model_dump_json(),
                image_paths=[rendered_page.path],
            )
        ]

    training_examples: list[TrainingExample] = []
    # Build one training example for every adjacent-page window.
    for page_a, page_b in consecutive_page_pairs(rendered_pages):
        # Label the first and second images as batch pages 1 and 2.
        pair_detections = [
            PageDetections(
                page=1,
                regions=regions_by_page.get(page_a.page_number, []),
            ),
            PageDetections(
                page=2,
                regions=regions_by_page.get(page_b.page_number, []),
            ),
        ]
        training_examples.append(
            TrainingExample(
                inputText=input_text,
                # Serialize both page labels as one JSON array.
                expectedOutput=json.dumps(
                    [detections.model_dump(mode="json") for detections in pair_detections]
                ),
                image_paths=[page_a.path, page_b.path],
            )
        )
    return training_examples



class AmbiguousMatchError(ValueError):
    pass



def locate_fragment_in_page(
    fragment_path: Path,
    page_path: Path,
    *,
    min_confidence: float = 0.98,
    ambiguity_margin: float = 0.02,
) -> tuple[int, int, int, int, float]:


    page_img = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
    fragment_img = cv2.imread(str(fragment_path), cv2.IMREAD_GRAYSCALE)

    if page_img is None:
        raise ValueError(f"could not read page image: {page_path}")
    if fragment_img is None:
        raise ValueError(f"could not read fragment image: {fragment_path}")

    fragment_height, fragment_width = fragment_img.shape[:2]
    page_height, page_width = page_img.shape[:2]

    if fragment_height > page_height or fragment_width > page_width:
        raise ValueError(
            f"fragment {fragment_path.name} ({fragment_width}x{fragment_height}) "
            f"is larger than page {page_path.name} ({page_width}x{page_height}) "
            f"-- likely a DPI mismatch between the fragment and this render"
        )

    result = cv2.matchTemplate(page_img, fragment_img, cv2.TM_CCOEFF_NORMED)
    _, best_val, _, best_loc = cv2.minMaxLoc(result)

    if best_val < min_confidence:
        raise ValueError(
            f"low-confidence match for {fragment_path.name} on {page_path.name}: "
            f"{best_val:.4f} < {min_confidence}"
        )

    masked_result = result.copy()
    best_x, best_y = best_loc
    mask_top = max(0, best_y - fragment_height // 2)
    mask_bottom = min(masked_result.shape[0], best_y + fragment_height // 2 + 1)
    mask_left = max(0, best_x - fragment_width // 2)
    mask_right = min(masked_result.shape[1], best_x + fragment_width // 2 + 1)
    masked_result[mask_top:mask_bottom, mask_left:mask_right] = -1.0  # TM_CCOEFF_NORMED floor

    _, second_val, _, second_loc = cv2.minMaxLoc(masked_result)

    if second_val >= best_val - ambiguity_margin:
        raise AmbiguousMatchError(
            f"{fragment_path.name} matches {page_path.name} at two locations with "
            f"near-tied confidence: {best_loc} @ {best_val:.4f} vs {second_loc} @ "
            f"{second_val:.4f} (margin {ambiguity_margin}) -- likely duplicated "
            f"content (e.g. a shared diagram printed for two questions); resolve "
            f"which occurrence is correct manually rather than trusting either"
        )

    left, top = best_loc
    right = left + fragment_width
    bottom = top + fragment_height
    return left, top, right, bottom, best_val


# reverse of normalized_box_to_pixels.
def pixels_to_normalized_box(
    pixel_box: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
) -> list[int]:
    left, top, right, bottom = pixel_box
    xmin = round(left * 1000 / width)
    ymin = round(top * 1000 / height)
    xmax = round(right * 1000 / width)
    ymax = round(bottom * 1000 / height)
    return [ymin, xmin, ymax, xmax]


def locate_fragment_across_pages(
    fragment_path: Path,
    rendered_pages: Sequence[RenderedPage],
    *,
    min_confidence: float = 0.98,
    ambiguity_margin: float = 0.02,
) -> tuple[RenderedPage, tuple[int, int, int, int], float]:
    last_error: Exception | None = None
    for rendered_page in rendered_pages:
        try:
            left, top, right, bottom, confidence = locate_fragment_in_page(
                fragment_path,
                rendered_page.path,
                min_confidence=min_confidence,
                ambiguity_margin=ambiguity_margin,
            )
        except (ValueError, cv2.error) as error:
            last_error = error
            continue
        return rendered_page, (left, top, right, bottom), confidence
    raise ValueError(
        f"no confident page match found for fragment {fragment_path}"
    ) from last_error


def build_training_examples_from_gold( #build TrainingExample
    data_path: Path,
    render_path: Path,
    *,
    input_text: str = vlm_prompt_text,
    dpi: int = 200,
    min_confidence: float = 0.98,
    ambiguity_margin: float = 0.02,
) -> list[TrainingExample]:
    training_examples: list[TrainingExample] = []
    for gold_path in sorted(data_path.rglob("gold.json")):
        paper_dir = gold_path.parent
        print(f"processing {paper_dir.name}...")
        with gold_path.open("r", encoding="utf-8") as gold_file:
            gold_data = json.load(gold_file)

        # corrected_data's gold.json is a flat list of one-region-per-entry
        # objects ({"page": N, "regions": [one_region]}), reviewed by hand
        # (or by another AI) against the actual paper "box_2d" and
        # "continuation" already come filled in, no fragment-to-page
        # matching needed. Multiple entries can share the same "page"
        # value (corrected_data gives one region per entry, not one entry
        # per physical page), so entries sharing a page get merged below.
        # The original data/ shape is a dict ({"regions": [...]}) with
        # "fragments" image-crop paths per region instead of a "box_2d",
        # handled unchanged further down.
        if isinstance(gold_data, list):
            paper_render_dir = render_path / paper_dir.relative_to(data_path)
            rendered_pages = render_pdf(
                paper_dir / "source.pdf",
                paper_render_dir,
                dpi=dpi,
            )
            pages_by_number = {
                rendered_page.page_number: rendered_page
                for rendered_page in rendered_pages
            }

            regions_by_page: dict[int, list[DetectedRegion]] = {}
            for entry in gold_data:
                page_number = entry["page"]
                for source_region in entry.get("regions", []):
                    detected_region = DetectedRegion(
                        paper=source_region.get("paper"),
                        type=cast(RegionType, source_region["type"]),
                        label=source_region.get("label"),
                        # corrected_data populates this directly (including
                        # "start"/"end" for a region split across pages);
                        # "single" only backstops an entry that omits it.
                        continuation=source_region.get("continuation") or "single",
                        
                        # Read the normalized coordinate field used by corrected data.
                        box_2d=source_region["box_2d"],
                        
                    )
                    regions_by_page.setdefault(page_number, []).append(
                        detected_region
                    )

            # a region referencing a page number with no matching rendered
            # page is a data error (typo, wrong paper) fail loudly rather
            # than silently dropping it.
            orphaned_pages = set(regions_by_page) - set(pages_by_number)
            if orphaned_pages:
                raise ValueError(
                    f"{gold_path}: region(s) reference page number(s) "
                    f"{sorted(orphaned_pages)} with no matching rendered "
                    f"page in {paper_dir / 'source.pdf'}"
                )


            # Include blank pages inside adjacent-page training windows.
            paper_examples = build_page_pair_training_examples(
                input_text=input_text,
                regions_by_page=regions_by_page,
                pages_by_number=pages_by_number,
            )
            print(f"  built {len(paper_examples)} example(s) for {paper_dir.name}")
            training_examples.extend(paper_examples)
  
            continue

        source_pdf_name = gold_data.get("source_pdf", "source.pdf")
        if not isinstance(source_pdf_name, str):
            raise ValueError(f"invalid source_pdf in {gold_path}")

        paper_render_dir = render_path / paper_dir.relative_to(data_path)
        rendered_pages = render_pdf(
            paper_dir / source_pdf_name,
            paper_render_dir,
            dpi=dpi,
        )
        regions_by_page: dict[int, list[DetectedRegion]] = {}

        for source_region in gold_data.get("regions", []):
            seen_fragment_bytes: set[bytes] = set()
            for fragment_value in source_region.get("fragments", []):
                if not isinstance(fragment_value, str):
                    raise ValueError(f"invalid fragment path in {gold_path}")

                fragment_path = paper_dir / Path(fragment_value)
                fragment_bytes = fragment_path.read_bytes()
                if fragment_bytes in seen_fragment_bytes:
                    continue
                seen_fragment_bytes.add(fragment_bytes)

                matched_page, pixel_box, _ = locate_fragment_across_pages(
                    fragment_path,
                    rendered_pages,
                    min_confidence=min_confidence,
                    ambiguity_margin=ambiguity_margin,
                )
                box_2d = pixels_to_normalized_box(
                    pixel_box,
                    width=matched_page.width,
                    height=matched_page.height,
                )
                detected_region = DetectedRegion(
                    paper=source_region.get("paper"),
                    type=cast(RegionType, source_region["type"]),
                    label=source_region.get("label"),
                    continuation=source_region.get("continuation"),
                    box_2d=box_2d,
                )
                regions_by_page.setdefault(matched_page.page_number, []).append(
                    detected_region
                )

        pages_by_number = {
            rendered_page.page_number: rendered_page
            for rendered_page in rendered_pages
        }

        # Reuse the corrected-data pair shape after fragment localization.
        paper_examples = build_page_pair_training_examples(
            input_text=input_text,
            regions_by_page=regions_by_page,
            pages_by_number=pages_by_number,
        )
        print(f"  built {len(paper_examples)} example(s) for {paper_dir.name}")
        training_examples.extend(paper_examples)

    print(f"built {len(training_examples)} example(s) total across all papers")
    return training_examples


# explicit guard for the scale assumption locate_fragment_in_page
# depends on, instead of inferring it from one dimension matching by
# coincidence. Checks both width and height against A4 at the given dpi.
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
MM_PER_INCH = 25.4


def expected_a4_pixel_size(dpi: int) -> tuple[int, int]:
    """Expected (width, height) in pixels for a portrait A4 page at dpi."""
    width = round(A4_WIDTH_MM / MM_PER_INCH * dpi)
    height = round(A4_HEIGHT_MM / MM_PER_INCH * dpi)
    return width, height


def verify_page_dpi(rendered_page: RenderedPage, *, dpi: int, tolerance_px: int = 3) -> None:
    """Raises ValueError if rendered_page's pixel dimensions don't match A4 at dpi.

    Call once per page, with the same dpi value passed to render_pdf,
    before running locate_fragment_in_page against that page -- confirms
    the assumption rather than assuming it from a single width match.
    """
    expected_width, expected_height = expected_a4_pixel_size(dpi)
    width_delta = abs(rendered_page.width - expected_width)
    height_delta = abs(rendered_page.height - expected_height)
    if width_delta > tolerance_px or height_delta > tolerance_px:
        raise ValueError(
            f"{rendered_page.path.name}: {rendered_page.width}x{rendered_page.height}px "
            f"doesn't match A4 @ {dpi}dpi ({expected_width}x{expected_height}px, "
            f"tolerance {tolerance_px}px) -- either this source page isn't A4, or "
            f"it wasn't rendered at dpi={dpi}"
        )

def calibrate_match_confidence(
    known_pairs: Sequence[tuple[Path, Path]],
) -> list[tuple[Path, float]]:
    scores: list[tuple[Path, float]] = []

    for fragment_path, page_path in known_pairs:
        page_img = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
        fragment_img = cv2.imread(str(fragment_path), cv2.IMREAD_GRAYSCALE)

        if page_img is None or fragment_img is None:
            scores.append((fragment_path, float("nan")))
            continue

        try:
            result = cv2.matchTemplate(
                page_img,
                fragment_img,
                cv2.TM_CCOEFF_NORMED,
            )
        except cv2.error:
            scores.append((fragment_path, float("nan")))
            continue

        _, max_val, _, _ = cv2.minMaxLoc(result)
        scores.append((fragment_path, max_val))

    return scores

# Copied from: experiments/gemini-3.5/src/gemini_paper_crop/images.py
@dataclass(frozen=True)
class CropArtifact:
    path: Path
    pixel_box: tuple[int,int,int,int]
    region_index: int


# Copied from: experiments/gemini-3.5/src/gemini_paper_crop/images.py
@dataclass(frozen=True)
class PageArtifacts:
    overlay_path: Path
    crops: list[CropArtifact]


# Copied from: experiments/gemini-3.5/src/gemini_paper_crop/images.py
def _safe_label(value: str | None) -> str:
    if value is None:
        return "unlabelled"
    cleaned = re.sub(r"[^a-zA-Z0-9.-]+", "-", value).strip("-")
    return cleaned or "unlabelled"


# to draw and output the detections made by VLM
def materialize_page_detections(
    page_path: Path,
    detections: PageDetections,
    output_dir: Path,
    *,
    padding: int = 12,
) -> PageArtifacts:
    overlays_dir = output_dir / "overlays"
    crops_dir = output_dir / "crops"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    crop_artifacts: list[CropArtifact] = []
    with Image.open(page_path) as source:
        page = source.convert("RGB")
        overlay = page.copy()
        draw = ImageDraw.Draw(overlay)

        for index, region in enumerate(detections.regions, start=1):
            pixel_box = normalized_box_to_pixels(
                region.box_2d,
                width=page.width,
                height=page.height,
                padding=padding,
            )
            region_dir = crops_dir / region.type
            region_dir.mkdir(parents=True, exist_ok=True)
            label = _safe_label(region.label)
            filename = (
                f"page-{detections.page}_region-{index}_"
                f"{label}_{region.continuation}.png"
            )
            crop_path = region_dir / filename
            page.crop(pixel_box).save(crop_path)
            crop_artifacts.append(
                CropArtifact(
                    path=crop_path,
                    pixel_box=pixel_box,
                    region_index=index,
                )
            )

            color = "#087f5b"
            draw.rectangle(pixel_box, outline=color, width=5)
            draw.text(
                (pixel_box[0] + 6, pixel_box[1] + 6),
                f"{index}: {region.type} {label}",
                fill=color,
                stroke_width=2,
                stroke_fill="white",
            )

        overlay_path = overlays_dir / page_path.name
        overlay.save(overlay_path)

    print(f"materialize_page_detections: saved overlay {overlay_path}, {len(crop_artifacts)} crop(s)")
    return PageArtifacts(overlay_path=overlay_path, crops=crop_artifacts)
#endregion functions

def file_exists_in_folder(fileName : str) -> bool:
    # Path to files/, relative to this script's own location
    # main/src/main/main.py -> parents[0]=main/src/main, [1]=main/src,
    # [2]=main so this lands on main/files/ regardless of
    # which folder the command actually got run from.
    source_path = folderPath / fileName
    print("finding :", source_path)
    pathExists = source_path.exists()
    print("File found" if pathExists else "File not found")
    return pathExists

def get_image_processor():
    global _image_processor
    if _image_processor is None:
        _image_processor = AutoImageProcessor.from_pretrained(
            defaultBaseModel, trust_remote_code=True
        )
    return _image_processor

def compute_expected_tokens(image_processor, image_path: Path) -> int:
    image = Image.open(image_path).convert("RGB")
    processed = image_processor(images=image, return_tensors="pt")
    grid_t, grid_h, grid_w = processed["image_grid_thw"][0]
    merge_size = getattr(image_processor, "merge_size", 2)
    return int((grid_t * grid_h * grid_w) // (merge_size ** 2))

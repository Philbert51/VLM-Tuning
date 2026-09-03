from pathlib import Path
from dataclasses import dataclass
from collections.abc import Sequence
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from dotenv import load_dotenv
import argparse
import pymupdf
import re
import tinker
import json
import os
import math

#initialization
#modifiable
folderPath = Path(__file__).resolve().parents[2] / "files" #desired folder path to scan the files, defaults to main/files
outputPath = Path(__file__).resolve().parents[2] / "output" #desired folder for output
max_output_tokens = 2000 #max tokens the VLM can generate per page, raise if responses get cut off
number_of_pages_per_sample = 1 #how many rendered pages get sent together in one VLM call
vlm_prompt_text = """You are given image(s) from a multi-page exam paper.

Output JSON only, matching this example. If given more than one image in this call, output one JSON object per image. You may output multiple objects if given multiple images.

Top-level fields:
- page_number (dtype: integer): this page's position among the image(s) given in this call, counting from 1.

Region types:
- cover_page: cover/title content (title, subject, date, instructions, marks table). Box its actual visual extent, same as any other region. If nothing else except cover page is present on that specific page, mark the whole box as [0,0,1000,1000]. Judge this per page, not across every image given in this call.
- mcq_question: Multiple Choice Question, a multiple-choice question.
- oe_question: Open Ended, an open-ended question.
- mcq_answer_key: Answer Key Multiple Choice Question, an answer-key entry for a multiple-choice question.
- oe_answer_key: Answer Key Open Ended, an answer-key entry for an open-ended question.

Each region also has:
- label (dtype: string | null): the printed question number/label, or null if there isn't one.
- continuation (dtype: enum["start", "middle", "end", "single"]): "single" when the region fits on one page. When a question or answer spans multiple pages, output one box per page with the same type, paper, and label, and set "start", "middle", or "end" in page order.
- box_2d (dtype: list of 4 integers): the region's bounding box as [ymin, xmin, ymax, xmax], normalized to integers 0-1000. (0,0) is the page's top-left corner, y increases downward, x increases rightward.
- If uncertain about the exact boundary, extend into surrounding blank whitespace rather than cutting into the question's text

<output_example>
{"page_number": 1, "regions": [{"type": "mcq_question", "label": "12", "continuation": "single", "box_2d": [120, 80, 340, 900]}]},
{"page_number": 2, "regions": [{"type": "oe_question", "label": "2", "continuation": "start", "box_2d": [20, 80, 340, 600]}]}
</output_example>

"""
#modifiable
continuation = Literal["start", "middle", "end", "single"]
samplingTemperature = 0.0 #0 = deterministic/repeatable output, higher = more varied output. this setting shouldn't be changed.
load_dotenv()
# defaultBaseModel : str = "Qwen/Qwen3.5-4B"
defaultBaseModel : str = "Qwen/Qwen3.6-35B-A3B"
PixelBox = tuple[int, int, int, int]
RegionType = Literal["cover_page", "mcq_question", "oe_question", "mcq_answer_key", "oe_answer_key"]

#initialization

def build_parser() -> argparse.ArgumentParser:
    # Defines the "--message" flag required, must be a string.
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:

    #testing delete later
    test_rendered_pages: list[RenderedPage] = [
        RenderedPage(page_number=2, path=outputPath / "page-2.png", width=1654, height=2339),
        RenderedPage(page_number=1, path=outputPath / "page-1.png", width=1654, height=2339)
        
    ]
    test_raw_responses: list[str] = ["""
     ```json
[
    {
        "page_number": 1,
        "regions": [
            {
                "type": "cover_page",
                "question_number": null,
                "fragment_index": 1,
                "needs_review": false,
                "box_2d": [
                    0,
                    0,
                    1000,
                    1000
                ]
            }
        ]
    },
    {
        "page_number": 2,
        "regions": [
            {
                "type": "mcq_question",
                "question_number": "1",
                "fragment_index": 1,
                "needs_review": false,
                "box_2d": [
                    195,
                    100,
                    350,
                    900
                ]
            },
            {
                "type": "mcq_question",
                "question_number": "2",
                "fragment_index": 1,
                "needs_review": false,
                "box_2d": [
                    395,
                    100,
                    550,
                    900
                ]
            },
            {
                "type": "mcq_question",
                "question_number": "3",
                "fragment_index": 1,
                "needs_review": false,
                "box_2d": [
                    595,
                    100,
                    800,
                    900
                ]
            }
        ]
    }
]
```
"""]

    #testing delete later

    return 0
    
    args = build_parser().parse_args(argv)

    # create folder and output folder, exist_ok make it so- 
    # that the program doesn't crash if it already exists
    folderPath.mkdir(parents=True, exist_ok=True)
    outputPath.mkdir(parents=True, exist_ok=True)

    # check if file exists
    fileName = args.filename
    filePath = folderPath / args.filename
    if not file_exists_in_folder(fileName) :
        raise FileNotFoundError(f"File {filePath} does not exist")


    # figure out the pdf's absolute width and height in pixel
    rendered_pages = render_pdf(pdf_path=filePath, output_dir=outputPath)

    # normalize it nothing to do here: the model outputs boxes already
    # normalized 0-1000 against the image's own edges, it never sees or
    # needs the real pixel width/height (established earlier this session).

    # send PDF to the VLM
    if not os.environ.get("TINKER_API_KEY"):
        raise ValueError("TINKER_API_KEY is missing. Add it to .env or export it.")

    # prompt sent to the VLM each call; boundary-uncertainty bullet below
    # biases toward overcrop over undercrop, but only into blank margin,
    # never into a neighboring question's content (added 2 Sept 2026,
    # reworded same day to close that overlap loophole)
   

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=defaultBaseModel)
    tokenizer = sampling_client.get_tokenizer()

    raw_responses = []

    # group rendered_pages into batches of number_of_pages_per_sample,
    # one VLM call happens per batch, not per page, when that's above 1
    page_batches = [
        rendered_pages[i:i + number_of_pages_per_sample]
        for i in range(0, len(rendered_pages), number_of_pages_per_sample)
    ]

    for page_batch in page_batches:
        # turn the prompt text into token ids the model reads
        prompt_tokens = tokenizer.encode(vlm_prompt_text)

        # start the model input with the prompt text tokens
        prompt = tinker.types.ModelInput.from_ints(prompt_tokens)

        # append every page image in this batch right after the text.
        # Order here is a best-effort guess, no example of ImageChunk
        # usage exists in the installed tinker package, and Tinker's own
        # docs were not consulted (declined earlier this session)
        for rendered_page in page_batch:
            image_bytes = rendered_page.path.read_bytes()
            prompt = prompt.append(tinker.types.ImageChunk(data=image_bytes, format="png"))

        # cap the response length, temperature=0 and should always be 0 because there can only be 1 correct output amongst the candidates
        sampling_params = tinker.types.SamplingParams(max_tokens=max_output_tokens, temperature=samplingTemperature)

        # send the request, blocks until the model responds
        future = sampling_client.sample(prompt=prompt, num_samples=1, sampling_params=sampling_params)
        response = future.result()

        # take the first (only) sample's token ids
        output_tokens = response.sequences[0].tokens

        # turn those token ids back into text, expected to be JSON
        raw_responses.append(tokenizer.decode(output_tokens))

    # change raw responses to array of objects holding rendered pages
    all_detections = [detection for raw_text in raw_responses for detection in extract_page_detections_batch(raw_text)]

    # match each detection set to its rendered page by page_number, not list
    # position -- zip() paired by index, which silently mismatches image and
    # detections if a batched VLM response ever returns page objects out of order
    pages_by_number = {rendered_page.page_number: rendered_page for rendered_page in rendered_pages}

    # draw overlays and save cropped regions for each page's detections
    for detections in all_detections:
        rendered_page = pages_by_number.get(detections.page_number)
        if rendered_page is None:
            print(f"no rendered page found for page_number={detections.page_number}")
        else :
            materialize_page_detections(rendered_page.path, detections, outputPath)


    # train the VLM

    # correct the vlm using loss value as -(log(p))

    # get the weights related to the API

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
    question_number: str | None = Field(
        default=None,
        max_length=32,
        description="Printed question number, or null for an MCQ answer table.",
    )
    fragment_index: int = Field(
        default=1,
        ge=1,
        description="Reading-order fragment number when an item spans pages.",
    )
    box_2d: list[int] = Field(
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
    page_number: int = Field(ge=1, description="One-based source PDF page number.")
    regions: list[DetectedRegion] = Field(default_factory=list, max_length=100)

#endregion classes

#region functions

# from render.py
def render_pdf(pdf_path: Path, output_dir : Path,dpi: int = 200,page_limit: int | None = None) -> list[RenderedPage]:
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
            rendered.append(
                RenderedPage(
                    page_number=page_index + 1,
                    path=path,
                    width=pixmap.width,
                    height=pixmap.height,
                )
            )
    return rendered


def create_training_client(base_model: str = defaultBaseModel, rank: int = 32) -> tinker.TrainingClient:
    #base_model = the parameter
    #if key not found
    if not os.environ.get("TINKER_API_KEY"):
        raise ValueError("TINKER_API_KEY is missing. Add it to .env or export it.")

    #initialize the client, already automatically grabs TINKER_API_KEY
    service_client = tinker.ServiceClient()

    # LoRA fine-tuning client for the given base model this is the
    # object forward_backward/optim_step calls run through during training.
    training_client = service_client.create_lora_training_client(
        base_model=base_model,
        rank=rank,
    )
    return training_client


def extract_page_detections(raw_text: str) -> PageDetections:
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
                return PageDetections.model_validate_json(object_text)

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

    return detections


#from images.py
def normalized_box_to_pixels(box_2d: Sequence[int], *, width: int, height: int, padding: int,) -> PixelBox:
    ymin, xmin, ymax, xmax = box_2d
    left = max(0, math.floor(xmin * width / 1000) - padding)
    top = max(0, math.floor(ymin * height / 1000) - padding)
    right = min(width, math.ceil(xmax * width / 1000) + padding)
    bottom = min(height, math.ceil(ymax * height / 1000) + padding)
    return left, top, right, bottom


def box_iou(box_a: Sequence[int, int, int, int], box_b: Sequence[int, int, int, int]) -> float:
    # Intersection over Union between two [ymin, xmin, ymax, xmax] boxes.
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


# Copied from: experiments/gemini-3.5/src/gemini_paper_crop/images.py
@dataclass(frozen=True)
class CropArtifact:
    path: Path
    pixel_box: PixelBox
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
            label = _safe_label(region.question_number)
            filename = (
                f"page-{detections.page_number}_region-{index}_"
                f"{label}_fragment-{region.fragment_index}.png"
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

            color = "#d62728" if region.needs_review else "#087f5b"
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

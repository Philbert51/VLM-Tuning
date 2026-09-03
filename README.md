# Teekathon: fine-tune a VLM to crop exam papers

Your task is to fine-tune a vision-language model using **Tinker**. Given a
page from a primary-school exam paper, the model should return every useful
region on that page as a labelled bounding box.

## Background

TEEBLOC turns full exam-paper PDFs into individual questions that can be shown
inside the TEEBLOC service. Instead of displaying an entire scanned page, the
service should be able to show just one question and, when needed, its matching
answer. Accurate crops also let TEEBLOC label, order, search, and connect each
question to the c
orrect answer key.

Exam papers contain several different kinds of useful regions. Use these exact
type names in the model output:

- `cover_page` — the page that identifies a booklet, paper, or section, such as
  Paper 1 or Section A
- `mcq_question` — one multiple-choice question, including its prompt, diagrams,
  and answer options
- `oe_question` — one open-ended question, including its diagrams, instructions,
  and sub-parts
- `mcq_answer_key` — the answer-key region containing answers to multiple-choice
  questions
- `oe_answer_key` — the worked answer or marking-scheme region for an open-ended
  question

MCQ means multiple-choice question, OE means open-ended question, and AK means
answer key.

The main goal is **100% performance on the supplied training papers**. This is
an intentional overfitting target for the hackathon, not a claim that the model
will generalise to new papers.

## Data

The repository contains 80 papers:

- 20 P5 Maths
- 20 P5 Science
- 20 P6 Maths
- 20 P6 Science

Run this once to check that all files are present:

```bash
uv sync
uv run teekathon-data validate
```

`dataset.json` lists the papers. Each paper directory contains:

```text
source.pdf          the original paper
gold.json           labels and paths for the reviewed crops
gold/               the correct crop images from TEEBLOC TREX
trex_manifest.json  correction history when available; normally not needed
```

The crops are the source of truth. Your first job is to match them back to the
rendered PDF pages and turn them into page numbers and bounding-box coordinates.
You may automate this, annotate manually, or combine both approaches.

TREX does not contain cover-page crops, so annotate cover pages directly from
the PDFs. Use the full-page box `[0, 0, 1000, 1000]` and record which
paper/section each cover belongs to.

## Expected model output

Use one JSON object per page and normalized integer coordinates from 0 to 1000:

```json
{
  "page": 12,
  "regions": [
    {
      "type": "oe_question",
      "paper": 1,
      "section": null,
      "label": "27",
      "continuation": "start",
      "box": [120, 70, 980, 930]
    }
  ]
}
```

The box order is `[y_min, x_min, y_max, x_max]`.
`paper`, `section`, and `label` may be `null` when they do not apply.

If a question or answer spans pages, output one box on each page with the same
type, paper and label. Set `continuation` to `start`, `middle`, or `end` in page
order. Use `single` when it fits on one page. A bounding box must never cross a
page boundary.

## What to build

1. Convert the supplied PDFs and gold crops into training examples.
2. Fine-tune a Tinker-supported VLM to produce the JSON above.
3. Provide an inference command that accepts a PDF and writes predictions.
4. Evaluate the fine-tuned model on all 80 supplied papers.

For this task, 100% means:

- every annotated region is returned with the correct type, paper and label;
- there are no extra regions; and
- every predicted box has at least 0.90 IoU with its annotation.

Your solution will also be evaluated on cost efficiency: achieve the required
accuracy with as little paid AI usage as possible. Include your actual total
spend and a short cost breakdown in your final results.

## AI credits

You may receive up to **US$80 in total AI-platform credits** for this project.
Before spending any credits, send a short proposal in the **Teekathon WhatsApp
group** stating:

- how much you are requesting;
- which service you will use; and
- how you expect to spend the credits.

Wait for approval before spending. You may request the budget in stages—for
example, US$40 initially and more later—but all approved requests combined
cannot exceed US$80. Tinker is our recommended service for this fine-tuning
task, although you may propose another AI platform.

You may also use the Teekathon WhatsApp group to clarify any project questions.

## What to submit

- the data-conversion and annotation code;
- the Tinker training code and configuration;
- the inference and evaluation code;
- the saved Tinker checkpoint/model path; and
- a short result showing the final training-set scores and how to reproduce them.

Keep the solution as simple as you can. You may choose any currently supported
vision model, training format, and sensible deterministic post-processing.

## Tinker references

- [Tinker quick start](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/)
- [Supported models](https://tinker-docs.thinkingmachines.ai/tinker/models/)
- [VLM training recipe](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/vlm-classifier/)
- [Vision input/rendering guide](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/rendering/)

## Optional Gemini baseline

[`experiments/gemini-3.5`](experiments/gemini-3.5/) contains an earlier
zero-shot Gemini 3.5 Flash cropping experiment. It is not part of the required
submission, but its PDF rendering, bounding-box parsing, crop generation, and
HTML reports may be useful as reference code or as a baseline for comparison.
Two existing runs are included there as viewable examples; new run outputs are
kept in that experiment's ignored `runs/` directory.


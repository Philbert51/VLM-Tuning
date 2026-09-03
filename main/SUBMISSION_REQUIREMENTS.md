# Teekathon — Hard Requirements Reference

## Region types — exact names required

README.md line 16: "Use these exact type names in the model output:"

- `cover_page` (line 18-19): the page that identifies a booklet, paper, or
  section, e.g. Paper 1 or Section A
- `mcq_question` (line 20-21): one multiple-choice question, including its
  prompt, diagrams, and answer options
- `oe_question` (line 22-23): one open-ended question, including its
  diagrams, instructions, and sub-parts
- `mcq_answer_key` (line 24-25): the answer-key region containing answers to
  multiple-choice questions
- `oe_answer_key` (line 26-27): the worked answer or marking-scheme region
  for an open-ended question

## Output schema

One JSON object per page (line 71), normalized integer coordinates 0-1000.

Per region (lines 74-90):
- `type` — one of the 5 exact type names above
- `paper` — nullable
- `section` — nullable, booklet label from the cover page (see organizer
  clarification below)
- `label` — nullable, printed question number as a string, e.g. `"27"`
- `continuation` — `start` | `middle` | `end` | `single`
- `box` — `[y_min, x_min, y_max, x_max]`, normalized 0-1000 (line 89)

`paper`, `section`, and `label` may be `null` when they don't apply (line 90).

## Cross-page splits

Lines 92-95: if a question or answer spans pages, output one box on each
page with the same `type`, `paper`, and `label`. `continuation` = `start` /
`middle` / `end` in page order, or `single` when it fits on one page. A box
must never cross a page boundary.

## Cover-page annotation rule

README.md line 65-66 (literal text, superseded below): TREX has no
cover-page crops — annotate cover pages directly from the PDFs. Use the
full-page box `[0, 0, 1000, 1000]` and record which paper/section each
cover belongs to. Read literally this is unconditional, but organizer
clarification (WhatsApp, 02/09/2026) overrides that for the mixed-content
case:

> README says cover pages always get the full-page box [0,0,1000,1000].
> What happens when a cover page shares the same physical page as real
> questions and other useful regions?
>
> Ah in that case it should not be full page, only the actual visual part

Actual rule:
- `cover_page` alone on its own page → full-page box `[0, 0, 1000, 1000]`
- `cover_page` sharing a page with real questions/other regions → box only
  its actual visual extent, not the full page
- most papers fall into the first case (organizer: "most of the time cover
  page will exist as its own full page"); the second is the exception,
  confirmed to occur (e.g. the Methodist Girls' School paper)

## Pipeline steps required

Lines 99-102:
1. Convert the supplied PDFs and gold crops into training examples.
2. Fine-tune a Tinker-supported VLM to produce the JSON schema above.
3. Provide an inference command that accepts a PDF and writes predictions.
4. Evaluate the fine-tuned model on all 80 supplied papers.

## Grading — 100% criteria

Lines 104-108:
- every annotated region returned with the correct `type`, `paper`, and
  `label`
- no extra regions
- every predicted box has at least 0.90 IoU with its annotation

`section` is not part of the grading criteria — absent from line 106.

## Cost efficiency

Lines 110-112: also evaluated on cost efficiency — achieve the required
accuracy with as little paid AI usage as possible. Actual total spend and a
short cost breakdown must be included in the final results.

## AI credits

Lines 114-127:
- up to US$80 total AI-platform credits
- before spending any credits, send a short proposal in the Teekathon
  WhatsApp group stating: how much is requested, which service, how the
  credits will be spent
- must wait for approval before spending
- may request in stages (e.g. $40 then more), but all approved requests
  combined cannot exceed $80
- Tinker is the recommended service; another platform may be proposed

## What must be submitted

Lines 133-137:
- the data-conversion and annotation code
- the Tinker training code and configuration
- the inference and evaluation code
- the saved Tinker checkpoint/model path
- a short result showing the final training-set scores and how to
  reproduce them

## Explicitly allowed flexibility (not a hard rule)

Lines 139-140: "Keep the solution as simple as you can. You may choose any
currently supported vision model, training format, and sensible
deterministic post-processing." Deterministic (non-VLM) post-processing is
explicitly permitted.

## Organizer clarification — `section` field

WhatsApp, 02/09/2026 18:14-18:15 (+65 8999 3032):
> for the section attribute, it should just refer to the "cover page" and
> extract from there. For example, the section for this paper is booklet A
>
> Therefore all questions that come after this cover page should be
> classified as booklet A

- `section` = the booklet label printed on the cover page (e.g. "Booklet A")
- applies to every question after that cover page, until the next cover
  page changes it
- confirmed: does not come from in-page headings like "Section A" printed
  mid-booklet — a different, unrelated use of the same word

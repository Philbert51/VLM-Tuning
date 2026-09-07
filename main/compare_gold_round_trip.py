from __future__ import annotations

import os
import re
import sys
import tempfile
import traceback
from collections import defaultdict
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PATH_TO_LOOK = r"path containing folder of the crops and source.pdf to generate visualized comparison" # used only for generating and helping build the formatted corrected_data
PATH_TO_LOOK = Path(PATH_TO_LOOK).resolve() # input
PROJECT_DIR = Path(__file__).resolve().parent  # input
SOURCE_PDF_NAME = "source.pdf"  # input
GOLD_FOLDER_NAME = "gold"  # input
OUTPUT_FOLDER_NAME = "comparison"  # output
FRAGMENT_EXTENSIONS = [".jpg", ".png"]  # input
IGNORED_FOLDER_NAMES = {"__pycache__"}  # input

sys.path.insert(0, str(PROJECT_DIR / "src"))

RENDER_DPI = 200  # input
MIN_CONFIDENCE = 0.90  # input
AMBIGUITY_MARGIN = 0.02  # input
MIN_SIFT_INLIERS = 8  # input
MAX_SIFT_AXIS_SCALE_RATIO = 1.25  # input
MAX_SIFT_OUTSIDE_MARGIN = 0.02  # input
SIFT_RANSAC_SEED = 0  # input
SHOW_COMPARISONS = False  # input
SCAN_WORKERS = min(4, os.cpu_count() or 1)  # input

from main.main import AmbiguousMatchError
from main.main import RenderedPage
from main.main import materialize_page_detections
from main.main import pixels_to_normalized_box
from main.main import render_pdf


@dataclass
class LooseDetectedRegion:
    """Same shape materialize_page_detections reads (.type/.label/
    .continuation/.box_2d), without DetectedRegion's strict RegionType
    literal -- this script only confirms a fragment's image content lines
    up with a position on the rendered page, so a fragment's folder name
    never needs to match a fixed set of known region types, and a fragment
    sitting outside the usual <region_type>/<paper>/<label> layout (e.g.
    dropped directly under gold/) still gets scanned and compared."""

    type: str
    label: str | None
    continuation: str
    box_2d: list[int]


@dataclass
class LoosePageDetections:
    page: int | None
    regions: list[LooseDetectedRegion]


def paper_directories() -> list[Path]:
    if (PATH_TO_LOOK / SOURCE_PDF_NAME).is_file():
        return [PATH_TO_LOOK]
    return sorted(
        path
        for path in PATH_TO_LOOK.iterdir()
        if path.is_dir() and path.name not in IGNORED_FOLDER_NAMES
    )


def natural_key(path: Path, paper_dir: Path) -> list[int | str]:
    relative_path = path.relative_to(
        paper_dir / GOLD_FOLDER_NAME
    ).as_posix()
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", relative_path)
    ]


def gold_fragments(paper_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in (paper_dir / GOLD_FOLDER_NAME).rglob("*")
            if path.is_file()
            and path.suffix.lower() in FRAGMENT_EXTENSIONS
        ),
        key=lambda path: natural_key(path, paper_dir),
    )


def locate_fragment_by_sift(
    fragment_path: Path,
    page_path: Path,
    fragment: np.ndarray,
    page: np.ndarray,
    fragment_keypoints: list,
    fragment_descriptors: np.ndarray | None,
    page_keypoints: list,
    page_descriptors: np.ndarray | None,
    *,
    min_confidence: float,
) -> tuple[int, int, int, int, float, int]:
    # keypoints/descriptors arrive precomputed computed once per fragment
    # and once per page (see locate_fragment_across_pages and
    # get_page_sift_features) instead of sift.detectAndCompute() re-running
    # full-page SIFT detection on the same fragment/page for every
    # (fragment, page) pair tested against it.
    if fragment_descriptors is None or page_descriptors is None:
        raise ValueError(f"no CV2 features found for {fragment_path}")

    matches = cv2.BFMatcher().knnMatch(
        fragment_descriptors,
        page_descriptors,
        k=2,
    )
    good_matches = [
        pair[0]
        for pair in matches
        if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
    ]
    if len(good_matches) < 4:
        raise ValueError(f"not enough CV2 feature matches for {fragment_path}")

    fragment_points = np.float32(
        [fragment_keypoints[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    page_points = np.float32(
        [page_keypoints[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    cv2.setRNGSeed(SIFT_RANSAC_SEED)
    homography, inlier_mask = cv2.findHomography(
        fragment_points,
        page_points,
        cv2.RANSAC,
        5.0,
    )
    if homography is None or inlier_mask is None:
        raise ValueError(f"no CV2 homography found for {fragment_path}")

    inlier_count = int(inlier_mask.sum())
    if inlier_count < MIN_SIFT_INLIERS:
        raise ValueError(
            f"not enough CV2 homography inliers for {fragment_path}: "
            f"{inlier_count} < {MIN_SIFT_INLIERS}"
        )

    confidence = float(inlier_mask.mean())
    feature_min_confidence = min(min_confidence, 0.65)
    if confidence < feature_min_confidence:
        raise ValueError(
            f"low-confidence CV2 feature match for {fragment_path}: "
            f"{confidence:.4f} < {feature_min_confidence}"
        )

    fragment_height, fragment_width = fragment.shape[:2]
    fragment_corners = np.float32(
        [
            [[0, 0]],
            [[fragment_width, 0]],
            [[fragment_width, fragment_height]],
            [[0, fragment_height]],
        ]
    )
    page_corners = cv2.perspectiveTransform(
        fragment_corners,
        homography,
    ).reshape(-1, 2)
    page_height, page_width = page.shape[:2]
    outside_x = page_width * MAX_SIFT_OUTSIDE_MARGIN
    outside_y = page_height * MAX_SIFT_OUTSIDE_MARGIN
    raw_left = float(page_corners[:, 0].min())
    raw_top = float(page_corners[:, 1].min())
    raw_right = float(page_corners[:, 0].max())
    raw_bottom = float(page_corners[:, 1].max())
    if (
        raw_left < -outside_x
        or raw_top < -outside_y
        or raw_right > page_width + outside_x
        or raw_bottom > page_height + outside_y
    ):
        raise ValueError(
            f"CV2 homography falls outside page bounds for {fragment_path}"
        )

    horizontal_scale = (
        np.linalg.norm(page_corners[1] - page_corners[0])
        + np.linalg.norm(page_corners[2] - page_corners[3])
    ) / (2 * fragment_width)
    vertical_scale = (
        np.linalg.norm(page_corners[3] - page_corners[0])
        + np.linalg.norm(page_corners[2] - page_corners[1])
    ) / (2 * fragment_height)
    if horizontal_scale <= 0 or vertical_scale <= 0:
        raise ValueError(f"invalid CV2 homography scale for {fragment_path}")
    axis_scale_ratio = max(horizontal_scale, vertical_scale) / min(
        horizontal_scale,
        vertical_scale,
    )
    if axis_scale_ratio > MAX_SIFT_AXIS_SCALE_RATIO:
        raise ValueError(
            f"distorted CV2 homography for {fragment_path}: "
            f"axis scale ratio {axis_scale_ratio:.3f} > "
            f"{MAX_SIFT_AXIS_SCALE_RATIO:.3f}"
        )

    left = max(0, int(np.floor(raw_left)))
    top = max(0, int(np.floor(raw_top)))
    right = min(page_width, int(np.ceil(raw_right)))
    bottom = min(page_height, int(np.ceil(raw_bottom)))
    if left >= right or top >= bottom:
        raise ValueError(f"invalid CV2 feature box for {fragment_path}")
    return left, top, right, bottom, confidence, inlier_count


def locate_fragment_in_page(
    fragment_path: Path,
    page_path: Path,
    fragment_img: np.ndarray,
    page_img: np.ndarray,
    *,
    min_confidence: float = 0.98,
    ambiguity_margin: float = 0.02,
) -> tuple[int, int, int, int, float]:
    # Mirrors main.locate_fragment_in_page's matching logic exactly, but
    # takes already-decoded grayscale arrays instead of re-reading
    # fragment_path/page_path from disk on every (fragment, page) pair
    # scan_fragments loads each page and fragment once and reuses the array.
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


def get_page_sift_features(
    page_number: int,
    page_img: np.ndarray,
    cache: dict[int, tuple[list, np.ndarray | None]],
    lock: Lock,
) -> tuple[list, np.ndarray | None]:
    # Computed once per page and reused by every fragment that reaches the
    # SIFT fallback for it, instead of every such fragment re-running
    # full-page SIFT detection on the same page image. The lock only guards
    # the check-and-populate step for a page not yet cached; SIFT detection
    # itself is deterministic for a given array, so a race here would only
    # ever risk redundant computation, never a wrong or inconsistent result.
    with lock:
        cached = cache.get(page_number)
        if cached is None:
            sift = cv2.SIFT_create()
            cached = sift.detectAndCompute(page_img, None)
            cache[page_number] = cached
        return cached


def locate_fragment_across_pages(
    fragment_path: Path,
    rendered_pages: list[RenderedPage],
    page_images: dict[int, np.ndarray],
    sift_cache: dict[int, tuple[list, np.ndarray | None]],
    sift_cache_lock: Lock,
    *,
    min_confidence: float,
    ambiguity_margin: float,
    stop_event: Event | None = None,
) -> tuple[RenderedPage, tuple[int, int, int, int], float]:
    # Decoded once here and reused for every page this fragment is tested
    # against, instead of locate_fragment_in_page/locate_fragment_by_sift
    # each re-reading fragment_path from disk on every page iteration.
    fragment_img = cv2.imread(str(fragment_path), cv2.IMREAD_GRAYSCALE)
    if fragment_img is None:
        raise ValueError(f"could not read fragment image: {fragment_path}")

    template_matches: list[
        tuple[float, RenderedPage, tuple[int, int, int, int]]
    ] = []
    last_error: Exception | None = None
    for rendered_page in rendered_pages:
        if stop_event is not None and stop_event.is_set():
            raise CancelledError("Fragment scanning stopped")
        try:
            left, top, right, bottom, confidence = locate_fragment_in_page(
                fragment_path,
                rendered_page.path,
                fragment_img,
                page_images[rendered_page.page_number],
                min_confidence=min_confidence,
                ambiguity_margin=ambiguity_margin,
            )
        except (ValueError, cv2.error) as error:
            last_error = error
            continue
        template_matches.append(
            (confidence, rendered_page, (left, top, right, bottom))
        )
    if template_matches:
        ranked_template_matches = sorted(
            template_matches,
            key=lambda match: match[0],
            reverse=True,
        )
        confidence, rendered_page, pixel_box = ranked_template_matches[0]
        if (
            len(ranked_template_matches) > 1
            and ranked_template_matches[1][0]
            >= confidence - ambiguity_margin
        ):
            second_confidence, second_page, _ = ranked_template_matches[1]
            raise AmbiguousMatchError(
                f"cross-page template tie for {fragment_path}: "
                f"page {rendered_page.page_number} @ {confidence:.4f} vs "
                f"page {second_page.page_number} @ {second_confidence:.4f}"
            )
        return rendered_page, pixel_box, confidence

    print(f"SIFT fallback triggered: {fragment_path}")
    # This fragment's own features are the same for every page it's tried
    # against below computed once here instead of once per page.
    sift = cv2.SIFT_create()
    fragment_keypoints, fragment_descriptors = sift.detectAndCompute(
        fragment_img,
        None,
    )
    sift_matches: list[
        tuple[int, float, RenderedPage, tuple[int, int, int, int]]
    ] = []
    for rendered_page in rendered_pages:
        if stop_event is not None and stop_event.is_set():
            raise CancelledError("Fragment scanning stopped")
        try:
            page_keypoints, page_descriptors = get_page_sift_features(
                rendered_page.page_number,
                page_images[rendered_page.page_number],
                sift_cache,
                sift_cache_lock,
            )
            (
                left,
                top,
                right,
                bottom,
                confidence,
                inlier_count,
            ) = locate_fragment_by_sift(
                fragment_path,
                rendered_page.path,
                fragment_img,
                page_images[rendered_page.page_number],
                fragment_keypoints,
                fragment_descriptors,
                page_keypoints,
                page_descriptors,
                min_confidence=min_confidence,
            )
        except (ValueError, cv2.error) as error:
            last_error = error
            continue
        sift_matches.append(
            (
                inlier_count,
                confidence,
                rendered_page,
                (left, top, right, bottom),
            )
        )
    if sift_matches:
        ranked_sift_matches = sorted(
            sift_matches,
            key=lambda match: (match[0], match[1]),
            reverse=True,
        )
        (
            inlier_count,
            confidence,
            rendered_page,
            pixel_box,
        ) = ranked_sift_matches[0]
        if len(ranked_sift_matches) > 1:
            (
                second_inlier_count,
                second_confidence,
                second_page,
                _,
            ) = ranked_sift_matches[1]
            if (
                second_inlier_count == inlier_count
                and second_confidence >= confidence - ambiguity_margin
            ):
                raise AmbiguousMatchError(
                    f"cross-page SIFT tie for {fragment_path}: "
                    f"page {rendered_page.page_number} with {inlier_count} "
                    f"inliers @ {confidence:.4f} vs page "
                    f"{second_page.page_number} with {second_inlier_count} "
                    f"inliers @ {second_confidence:.4f}"
                )
        return rendered_page, pixel_box, confidence

    raise ValueError(
        f"no confident page match found for {fragment_path}"
    ) from last_error


def scan_fragments(
    fragments: list[Path],
    rendered_pages: list[RenderedPage],
    page_images: dict[int, np.ndarray],
    sift_cache: dict[int, tuple[list, np.ndarray | None]],
    sift_cache_lock: Lock,
    *,
    min_confidence: float,
    ambiguity_margin: float,
    workers: int = SCAN_WORKERS,
) -> list[tuple[RenderedPage, tuple[int, int, int, int], float]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if not fragments:
        return []

    stop_event = Event()
    matches = {}
    previous_cv_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)

    def scan(fragment_path: Path):
        if stop_event.is_set():
            raise CancelledError("Fragment scanning stopped")
        print(f"Scanning: {fragment_path}", flush=True)
        return locate_fragment_across_pages(
            fragment_path,
            rendered_pages,
            page_images,
            sift_cache,
            sift_cache_lock,
            min_confidence=min_confidence,
            ambiguity_margin=ambiguity_margin,
            stop_event=stop_event,
        )

    try:
        with ThreadPoolExecutor(max_workers=min(workers, len(fragments))) as pool:
            pending = {}
            try:
                for index, fragment_path in enumerate(fragments):
                    pending[pool.submit(scan, fragment_path)] = index
                for future in as_completed(pending):
                    index = pending[future]
                    try:
                        matches[index] = future.result()
                    except Exception:
                        # This one fragment's own match failed (bad crop,
                        # unmatched screenshot, corrupt image, no confident
                        # match on any page, etc.) logged in full so it
                        # can be diagnosed later, without losing every other
                        # fragment's already-completed result.
                        print(
                            f"CRASH scanning fragment {fragments[index]}:",
                            flush=True,
                        )
                        print(traceback.format_exc(), flush=True)
                        matches[index] = None
                    print(
                        f"Scanned {len(matches)}/{len(fragments)}: "
                        f"{fragments[index]}",
                        flush=True,
                    )
            except BaseException:
                stop_event.set()
                for future in pending:
                    future.cancel()
                raise
    finally:
        cv2.setNumThreads(previous_cv_threads)

    return [matches[index] for index in range(len(fragments))]


def compare_gold_round_trip(
    paper_dir: Path,
    *,
    dpi: int = RENDER_DPI,
    min_confidence: float = MIN_CONFIDENCE,
    ambiguity_margin: float = AMBIGUITY_MARGIN,
    show: bool = SHOW_COMPARISONS,
    workers: int = SCAN_WORKERS,
) -> list[Path]:
    if workers < 1:
        raise ValueError("workers must be positive")
    comparison_dir = paper_dir / OUTPUT_FOLDER_NAME
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_records: list[
        tuple[int, Path, Image.Image, Image.Image, list[int], float]
    ] = []

    with tempfile.TemporaryDirectory(prefix="teekathon-round-trip-") as temporary:
        temporary_dir = Path(temporary)
        rendered_pages = render_pdf(
            paper_dir / SOURCE_PDF_NAME,
            temporary_dir / "rendered_pages",
            dpi=dpi,
        )

        # Decode each rendered page once up front and reuse the same array
        # across every fragment scan locate_fragment_in_page/
        # locate_fragment_by_sift previously re-read + re-decoded a page from
        # disk for every single fragment tested against it (fragments x
        # pages reads instead of just pages reads). Read-only arrays shared
        # across scan_fragments' worker threads; matchTemplate/SIFT never
        # mutate their inputs, so no locking is needed.
        page_images: dict[int, np.ndarray] = {}
        for rendered_page in rendered_pages:
            page_image = cv2.imread(str(rendered_page.path), cv2.IMREAD_GRAYSCALE)
            if page_image is None:
                raise ValueError(f"could not read page image: {rendered_page.path}")
            page_images[rendered_page.page_number] = page_image

        # Populated lazily only fragments that actually reach the SIFT
        # fallback ever compute a page's features, but every such fragment
        # after the first for a given page reuses the cached result instead
        # of re-running full-page SIFT detection.
        sift_cache: dict[int, tuple[list, np.ndarray | None]] = {}
        sift_cache_lock = Lock()

        fragments = gold_fragments(paper_dir)
        matches = scan_fragments(
            fragments,
            rendered_pages,
            page_images,
            sift_cache,
            sift_cache_lock,
            min_confidence=min_confidence,
            ambiguity_margin=ambiguity_margin,
            workers=workers,
        )
        for index, (fragment_path, match) in enumerate(
            zip(fragments, matches), start=1
        ):
            if match is None:
                # Already logged inside scan_fragments this fragment's
                # own scan failed, so there's nothing to build a comparison
                # figure from; move on to the next fragment.
                continue
            try:
                rendered_page, pixel_box, confidence = match
                box_2d = pixels_to_normalized_box(
                    pixel_box,
                    width=rendered_page.width,
                    height=rendered_page.height,
                )
                print(
                    f"Matched: page={rendered_page.page_number}, "
                    f"pixel_box={pixel_box}, box_2d={box_2d}",
                    flush=True,
                )
                # gold_fragments() already scans every png/jpg recursively under
                # gold/ regardless of depth these next two lines are read only
                # for the comparison figure's crop folder name and title text,
                # never for the match itself, so any depth under gold/ (including
                # a file sitting directly in gold/, with nothing to read for
                # region_type/label) is handled without crashing.
                fragment_parts = fragment_path.relative_to(
                    paper_dir / GOLD_FOLDER_NAME
                ).parts
                region_type = fragment_parts[0]
                label_value = fragment_parts[-2] if len(fragment_parts) >= 2 else None
                label = (
                    None if label_value in (None, "unlabelled") else label_value
                )
                detections = LoosePageDetections(
                    page=rendered_page.page_number,
                    regions=[
                        LooseDetectedRegion(
                            type=region_type,
                            label=label,
                            continuation="single",
                            box_2d=box_2d,
                        )
                    ],
                )
                artifacts = materialize_page_detections(
                    rendered_page.path,
                    detections,
                    temporary_dir / "materialized" / f"{index:02d}",
                    padding=0,
                )
                with Image.open(fragment_path) as gold_image:
                    actual_gold = gold_image.convert("RGB")
                with Image.open(artifacts.crops[0].path) as translated_image:
                    translated_gold = translated_image.convert("RGB")
                comparison_records.append(
                    (
                        rendered_page.page_number,
                        fragment_path,
                        actual_gold,
                        translated_gold,
                        box_2d,
                        confidence,
                    )
                )
            except Exception:
                print(
                    f"CRASH building comparison figure for {fragment_path}:",
                    flush=True,
                )
                print(traceback.format_exc(), flush=True)
                continue

    comparison_paths: list[Path] = []
    records_by_page: dict[
        int,
        list[
            tuple[int, Path, Image.Image, Image.Image, list[int], float]
        ],
    ] = defaultdict(list)
    for record in comparison_records:
        records_by_page[record[0]].append(record)

    for page_number, sheet_records in sorted(records_by_page.items()):
        figure, axes = plt.subplots(
            len(sheet_records),
            2,
            figsize=(14, 4 * len(sheet_records)),
            squeeze=False,
        )
        figure.patch.set_facecolor("black")
        for row, (
            _,
            gold_path,
            actual_gold,
            translated_gold,
            box_2d,
            confidence,
        ) in enumerate(sheet_records):
            axes[row][0].imshow(actual_gold)
            axes[row][1].imshow(translated_gold)
            axes[row][0].set_facecolor("black")
            axes[row][1].set_facecolor("black")
            axes[row][0].set_title(
                f"Actual gold crop\n{gold_path.relative_to(paper_dir)}",
                color="white",
            )
            axes[row][1].set_title(
                f"Normalized round trip\nbox={box_2d}, confidence={confidence:.4f}",
                color="white",
            )
            axes[row][0].axis("off")
            axes[row][1].axis("off")

        figure.tight_layout()
        comparison_path = comparison_dir / f"page-{page_number}.png"
        figure.savefig(
            comparison_path,
            dpi=150,
            bbox_inches="tight",
            facecolor="black",
        )
        comparison_paths.append(comparison_path)

    if show:
        plt.show()
    else:
        for figure_number in plt.get_fignums():
            plt.close(plt.figure(figure_number))

    return comparison_paths


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


def main() -> None:
    for paper_dir in paper_directories():
        print(f"Processing: {paper_dir}")

        # Mirror every print() (including any crash logged below) into a
        # log file next to this paper's data keeps the full run
        # readable back later even past the terminal's scrollback.
        log_path = paper_dir / "compare_gold_round_trip_log.txt"
        log_file = open(log_path, "a", encoding="utf-8")
        original_stdout = sys.stdout
        sys.stdout = _Tee(original_stdout, log_file)
        print(f"logging to {log_path}")

        try:
            comparison_paths = compare_gold_round_trip(
                paper_dir,
                dpi=RENDER_DPI,
                min_confidence=MIN_CONFIDENCE,
                ambiguity_margin=AMBIGUITY_MARGIN,
                show=SHOW_COMPARISONS,
                workers=SCAN_WORKERS,
            )
            for comparison_path in comparison_paths:
                print(comparison_path)
        except Exception:
            print(f"CRASH processing paper {paper_dir}:", flush=True)
            print(traceback.format_exc(), flush=True)
        finally:
            sys.stdout = original_stdout
            log_file.close()
    print("Finished.")


if __name__ == "__main__":
    main()

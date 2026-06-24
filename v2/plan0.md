# Refactoring Plan

The pipeline consists of four stage modules (`stage0`–`stage3`), a small device/init package, and a notebook (`pipeline.ipynb`) that orchestrates everything. The code is correct and functional; the goal of all rounds below is to make it easier to read and maintain **without changing any behavior**.

---

## Round 1 — Extract General Math and Remove Duplicate Code

**General math module.** Create a new module for geometry and signal-processing primitives that have no pipeline-specific logic. Two clear candidates live buried inside stage files right now:

- The polar coordinate transforms (converting an image between Cartesian and polar representations centered on a given circle) appear in both stage 0 and stage 3 with slightly different implementations. A single, well-parameterised pair of functions covers both uses.
- The circumcenter computation (given three points on a circle, find the center) is a self-contained geometric primitive currently hidden inside the moon-refinement function.

**Shared utility module.** Several short helper functions are copy-pasted across stage files with identical or near-identical bodies: a function that loads an image as a grayscale tensor, a function that applies a shift+rotation transform to a single image, a function that computes a weighted average of a stack of images, and a helper that computes the median value inside the moon disk. Move one canonical copy of each into a shared utilities module, then replace every duplicate with an import.

Both parts of this round are purely mechanical — no logic changes, just moving code to where it belongs.

---

## Round 2 — Consolidate Registration / Discrepancy Logic

The Fourier-based alignment scoring function appears in two stage files. The two versions differ only in whether they accept an optional validity mask. Merge them into a single function that accepts the mask as an optional argument (defaulting to "use all pixels"). Similarly, the grid-search registration loop (scan candidate shifts/rotations, score each, pick the best) is written out almost identically in two separate stages. Extract it into a shared function that accepts the scoring function as a parameter so the two stages can reuse it. This round also consolidates the transform-point batching helpers that accompany the registration code. Where these helpers use the polar transform, they should now call the function extracted in Round 1.

---

## Round 3 — Break Up Long Functions

Several functions are too long to read comfortably — they combine several distinct conceptual steps in one body. Each such function should be split into smaller private helpers with descriptive names, while keeping the original public function as a thin wrapper that calls them in order. The main candidates:

- The moon-edge refinement function in stage 0: separate the gradient extraction, the clustering/filtering steps, and the call into the circumcenter helper (already extracted in Round 1).
- The intra-exposure registration function in stage 0: separate the pair-scoring loop, the triplet-consistency check, and the result-filtering step.
- The pose optimization function in stage 1: separate the initial pose estimation, the loss definition, and the optimizer loop.
- The radial normalization function in stage 3: separate the polar-transform and extrapolation step, the tone-mapping/percentile step, and the blending step.

---

## Round 4 — Clean Up Naming and Magic Numbers

After the structural changes above, do a naming pass:

- Rename any function or variable whose name does not clearly describe what it does. Focus on internal helpers with cryptic names derived from implementation details rather than purpose.
- Gather the algorithm tuning constants (sector count, iteration counts, optimizer learning rate, patch sizes, sigma values, etc.) that are currently scattered as bare numeric literals across the stage files. Group them near the top of each file as named constants, and add a one-line comment to each explaining what it controls and why the current value was chosen (if known).
- Standardize the naming convention across module-level functions: currently some carry a `stage0_` / `stage1_` prefix and others do not; pick one convention and apply it consistently.

---

## Round 5 — Separate Concerns

Each stage module currently mixes computation code with notebook-display helpers (functions that produce GIFs, HTML grids, symlinks for Jupyter). Move the display helpers out of the stage modules and into a separate module (or a `display.py` file). The stage modules should then contain only computation; the display helpers import from the stages but not the other way around. Also, the large context object in stage 3 accumulates many fields of mixed character (inputs, intermediate arrays, geometry parameters, output paths). Document each field with a one-line comment describing what it holds and at which step it is populated, so a reader can understand the object's lifecycle without tracing execution.

---

## Notes

- Each round should leave the test output identical to before it started. The easiest check is to re-run the notebook on the existing pickles and compare the final images.
- Rounds 1 and 2 have the biggest payoff for the least risk and should be done first.
- Round 5 is the most subjective; if the display-helper separation feels like it adds complexity rather than removing it, it can be skipped or scoped down.

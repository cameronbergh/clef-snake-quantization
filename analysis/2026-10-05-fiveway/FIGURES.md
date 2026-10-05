# Five-way publication figures and provenance

These exports derive only from the completed [five-way discovery results](../../data/2026-10-05-fiveway/results.json). They add Q2_K to the original comparison; they do not add new environment seeds. The original dataset, uncertainty calculation and figures remain unchanged.

| Export | Intended use |
|---|---|
| [Wide PNG](quantization-snake-results.png) / [SVG](quantization-snake-results.svg) | README, desktop viewing or slides |
| [Stacked PNG](quantization-snake-results-mobile.png) / [SVG](quantization-snake-results-mobile.svg) | Narrow screens or portrait sharing |
| [Uncertainty note](UNCERTAINTY.md) / [JSON](uncertainty.json) | Exact numerical results, methods and exploratory tests |
| [Provenance](provenance.json) | Source, generator and export SHA256 hashes; package versions |

## Caption

CLEF-Flash Snake: 75 games across 15 paired environment seeds. Dots show colliding games, diamonds show model means, and lines connect outcomes for the same seed. Repeated scores overlap. The triangle marks the one IQ2_M game still alive at the 500-successful-move cap: 36 food on seed round 13 (`4a4af34fab25894939f1ec37802af6db`). Its observed score is retained; its ultimate score is unknown. The 38-food IQ2_M game collided and is a separate observation.

Intervals show the 2.5th and 97.5th percentiles of 100,000 resamples of entire paired seed rows, using NumPy PCG64 seed 20261005. They are unadjusted, descriptive intervals from an adaptive discovery experiment. BF16 uses a different runtime from the quantized models, and IQ2_M/Q2_K were evaluated later. The figures do not establish a monotonic, causal or general reasoning benefit. See the [uncertainty note](UNCERTAINTY.md) for the four-comparison Holm family and further limitations.

## Recreate

With the analysis dependencies in [requirements-analysis.txt](../../requirements-analysis.txt), run from the repository root into a new directory:

```sh
python scripts/analyze.py data/2026-10-05-fiveway/results.json runs/fiveway-analysis --publication-figures
```

The optional publication flag adds the cap marker, wide layout and stacked layout. The default historical layout remains available without the flag. Seed/model score validation now checks each paired record directly, rather than only comparing marginal score distributions. No model code, inference, GPU or network is used.

## Verification

- All 75 unique seed/model records, means, win/tie/loss counts, the cap, exact sign-flip tests and four-comparison Holm adjustment were independently recalculated from the round records.
- Both PNG layouts were visually inspected; labels, cap marker, uncertainty intervals and caveats are readable with no overlap.
- A second generation produced byte-identical uncertainty JSON/Markdown and all four figure files in this environment.
- Recalculating the original 60-game data with default options produced JSON and Markdown identical to the preserved historical uncertainty artifacts.

Reproducibility of figure bytes is scoped to the recorded Python/NumPy/Matplotlib environment. Rendering or font changes in other environments may change figure hashes while preserving the numerical results.

# Dead ends and decisions

A running log of what was tried, what was rejected, and the number or reason
that settled it. Kept so that "why did you do it that way?" has an answer.

## Rejected before building

- **Streaming drift detectors (ADWIN, DDM, Page-Hinkley).** Structurally
  impossible on this corpus: ACS is annual and cross-sectional, giving at most
  5 time points per state. There is no sequential error stream to run them on.
  Reporting "detection delay in days" here would be fabricated.
- **Seasonality analysis.** Annual survey data has no intra-year cycle.
- **A second corpus (Freddie Mac, Lending Club, MIMIC).** Acquisition alone
  exceeds the time budget; PhysioNet credentialing does not clear in two weeks.
  Depth on one corpus with real ground truth beats breadth without it.
- **Simulated label-arrival delay.** Self-refuting: the project's thesis is that
  results must not rest on effects the author injected, and a seeded delay
  distribution is an injected effect. Label-blindness is enforced structurally
  instead — detectors receive no `y` column at all.
- **Auto-retraining on alert.** Production teams deliberately do not do this
  without a human. It would also require labels that the project has just spent
  its length establishing are unavailable at decision time.
- **More divergence measures** (Wasserstein, Jensen-Shannon, Hellinger, energy
  distance). All are marginal tests; a seventh would land in the same cluster as
  the first three and add a row to the leaderboard, not a finding.

## Decisions made during the build

- **Fixed n per window (5,000).** PSI and KS are both functions of sample size.
  California has roughly 100x the rows of Wyoming, so unequal windows would have
  silently confounded every detector comparison with population size.
- **Reference cell drawn at 25,000 rather than 5,000.** Its 20% holdout is then
  exactly 5,000 rows, matching window size. A 1,000-row reference against
  5,000-row windows hands every two-sample test an asymmetry unrelated to drift.
- **Python 3.9 rather than 3.14.** `import scipy.stats` stalls indefinitely in
  `dlopen` of the `_biasedurn` extension on macOS 12 with scipy 1.18 / cp314;
  confirmed with `faulthandler` and reproduced outside any sandbox. sklearn
  imports scipy.stats, so the whole stack was blocked. Pinned versions in
  `requirements.txt`.
- **Raw census downloads deleted per cell.** The full 2014-2018 PUMS pull is
  tens of GB; only 5,000 rows per cell are ever needed.

## Build environment notes (recorded during the run)

- **First `dlopen` of each newly installed C extension is very slow on this
  machine.** macOS evaluates the code signature of each binary on first load;
  with `trustd` saturated, importing the numpy/scipy/sklearn/pyarrow stack took
  minutes rather than seconds. It is a one-time, per-binary cost — subsequent
  process starts are fast — but it dominated the first hour of the build.
  Diagnosed with `faulthandler.dump_traceback_later`, which pointed at
  `create_module` inside the extension loader rather than at any project code.
  Running several Python processes that import the same fresh extensions
  concurrently makes it worse, not better; they contend rather than share.
- **Measured corpus throughput: roughly 2 minutes per (state, year) cell**,
  download plus parse plus subsample. The raw ACS file for a large state is
  100-350 MB, which is why `_purge_cache()` runs after every cell.

## Corpus exclusions

Cells with fewer than `WINDOW_N` rows after the folktables adult filter are
excluded rather than compared at a smaller n. Comparing a 3,600-row window
against a 5,000-row reference would bias every KS and PSI value by population
size instead of by drift, which is precisely the confound the fixed-n design
exists to remove.

Observed so far: AK (3,643 rows), DE (4,619). Small-population states are
expected to drop out systematically. Every exclusion and its row count is
recorded in `data/build_log.csv` and listed by name in the README.

**Note on a deviation NOT taken.** Lowering `WINDOW_N` to 4,000 would recover
most excluded states. That change was considered and rejected *after* seeing
which states failed, which is exactly when it becomes a garden-of-forking-paths
decision. If the threshold is ever revisited, both the original and the revised
corpus must be reported side by side.

## Final corpus composition (2014 + 2018)

104 (state, year) cells attempted, 90 materialised, 45 per year — symmetric
across years, which matters: an exclusion that applied to one year but not the
other would confound temporal comparisons with sample availability.

Two distinct exclusion causes, kept separate rather than lumped as "missing":

1. **Below the fixed-n floor** (6 states x 2 years = 12 cells): AK, DE, ND, SD,
   VT, WY. Row counts after the folktables adult filter run 3,064-4,899 against
   a required 5,000. These are the small-population states and the exclusion is
   systematic, not random — the corpus therefore under-represents low-population
   states, and any claim about "all states" would be false.
2. **Loader failure** (DC, both years): `folktables` raises a bare
   `AssertionError` reading the DC person file. Not a size problem — DC has
   enough rows. Cause not investigated; it is one cell in each year and chasing
   a third-party loader bug was not worth the corpus time.

Net: 90 windows, of which 1 is the reference cell, leaving 89 deployment
windows — against the ~250 anticipated in the pre-registration, which assumed
five survey years. Years 2015-2017 were not downloaded: measured throughput was
about 2 minutes per cell and the machine was heavily loaded, so the corpus was
capped at the two years giving maximum temporal separation (2014 vs 2018) plus
full spatial coverage. The code takes years as arguments (`python -m
modelwatch.data 2015 2016 2017`) and the corpus extends without modification.

**Consequence to report, not hide:** 89 windows gives materially wider
confidence intervals than 250 would. Detectors separated by a few points of
PR-AUC cannot be honestly ranked against each other at this N.

## Adversarial QA pass — six confirmed defects

An independent multi-lens review (statistical correctness, label leakage,
reproducibility, edge cases, library API misuse, claims-vs-code, experimental
design) produced 41 raw findings; 16 went to adversarial verifiers instructed to
refute them. Six survived, every one reproduced by running code. All are fixed.

### 1. The headline result was an artifact of a smoothing constant (major)

`PSI_EPSILON = 1e-6` was 200x below the sample resolution `1/n = 2e-4`. In
`psi_categorical`, a category present in one sample and absent from the other is
clipped to that floor and contributes roughly `(1/n) * ln(1/(n*eps))`, which
diverges as eps shrinks. ACS OCCP has ~465 levels at n=5,000, so dozens of
categories are one-sided by pure sampling. Measured on a single real A/A split:
**56% of OCCP's PSI value and 58% of POBP's came from zero-count cells** rather
than from any shift in observed frequencies.

The project's most-quoted claim — "PSI > 0.2 sits below the no-drift noise floor
and fires on 100% of samples where nothing happened" — was therefore produced by
this constant, and the README attributed it to "sample size and bin count",
which was the wrong cause. The sweep (`modelwatch/eps_sweep.py`, 200 A/A splits):

| smoothing floor | no-drift median max-PSI | fires at PSI>0.2 |
|---|---|---|
| 1e-6 (original) | 0.293 | 100.0% |
| 1e-5            | 0.226 |  98.5% |
| 1e-4            | 0.156 |   0.0% |
| 0.5/n (current) | 0.156 |   0.0% |
| 1e-3            | 0.072 |   0.0% |

Fixed by `detectors.resolution_eps` = `0.5/n` — half a count, the smallest
quantity the sample could have resolved. Deliberately NOT fixed by raising the
constant: at 1e-3 the floor clips genuine low-count cells on both sides and
destroys real signal along with the artifact.

Two things are worth recording precisely, because they bound the damage:

- `psi_numeric` is provably insensitive to eps at these settings — with 10
  quantile bins at n=5,000 the clip never binds, measured difference exactly
  0.0. The constant's justification was documented on the function where it has
  no effect, and undocumented on the one where it decided the answer.
- The **second** failure mode survives untouched. Real-window max-PSI is driven
  by POBP, where only ~5% of the value is eps-derived: median 5.32 at 1e-6
  versus 5.14 at the resolution floor. So "drift is ubiquitous, harm is not",
  and the entire detector leaderboard, are unaffected.

The wrong number is kept and measured rather than deleted. A monitoring project
that quietly reversed its own headline would be making exactly the mistake it
was built to expose.

### 2. The published model hash was environment-dependent (major)

scikit-learn stores `_BinMapper(n_threads=_openmp_effective_n_threads())` on the
fitted estimator, and joblib pickles it — so `sha256(model.joblib)` changes with
the machine's core count for a bit-identical model. At `OMP_NUM_THREADS=2` the
digest matched the published pin; at `1` it did not, while `n_iter_`, holdout
AUC and the summed scores were identical. A CI runner would have "failed" the
integrity check on a correct model. Now hashed over the predictor nodes, bin
thresholds and baseline prediction — verified bit-equal across 1/2/4 threads.
The file digest is retained as `file_sha256`, labelled environment-dependent.

### 3. Bootstrap seeding — two defects, one found by QA and one by follow-up

**(a) Intervals depended on corpus composition.** A single RNG was threaded
sequentially through the loop over cells, so each window's CI was a function of
how many cells sorted before it. Extending the corpus from 2 years to 5 inserts
`*_2015`-`*_2017` cells ahead of the 2018 ones and would have silently rewritten
every already-published interval — and could flip the `harmful` label on a
borderline window. Now keyed per cell via `corpus.cell_generator`, which derives
a stream from SHA-256 of `(state, year, SEED)`. Never key it on the loop index,
and do not use `SeedSequence.spawn` either: spawned children are ordered, which
is the same positional coupling wearing a different hat. The full 256-bit digest
is handed to `SeedSequence` as entropy rather than truncated to 64 bits.

**(b) One holdout resample was shared by every window.** The first fix left
`delta_boot = win_boot - holdout_boot` with a single `holdout_boot` vector
computed once outside the loop — originally an optimisation ("saves ~250x the
work with identical semantics"). The semantics are not identical once the CIs
are aggregated. `harmful` is defined as `delta_auc <= -0.02 AND ci_hi < 0`, and
the headline base rate is the mean of that label over all windows, so a single
unlucky holdout realisation shifts every window's interval the same direction
and the base-rate CI understates the true uncertainty. Both sides are now
resampled from the window's own stream, which makes windows mutually
independent. This costs a second bootstrap pass per window and is cheap:
`fast_auc` is rank-based. It is also the more literal reading of the
pre-registration, which specifies "resampling the window and the reference
holdout independently at their fixed sizes".

### 4. `silent_failure_rate` was reported against the wrong denominator (minor)

`benchmark.score_binary` defines it as `fn/n` over all windows, and
`PREREGISTRATION.md` pre-registers it as a quadrant rate, so the metric is
correct. The model card rendered it as "the monitor misses X% of all windows
that were materially degraded", which is `fn/(fn+tp)`. The two differ by the
base rate. Invisible in the artifact only because `fn = 0`. The card now states
both denominators explicitly.

### 5. `build_log.csv` was destructive (minor) — and this one actually bit

`data.main` overwrote the log with only the years processed in that invocation.
It is the sole tracked record of the raw row count each 5,000-row subsample was
drawn from, and the subsample is a function of that count. The 2015-2017
extension run destroyed the 2014/2018 provenance before the fix landed; it was
recovered from git history and merged. Writes are now additive, and `available`
is carried forward for cached cells whose early return cannot know it.

### 6. Refuted findings

Ten claims did not survive verification, including several plausible ones: label
leakage into the detector battery (the domain classifier selects `C.FEATURES`,
which excludes `y`), invalid reuse of a single holdout bootstrap vector (valid
marginally; it correlates windows' CIs, which nothing aggregates), and
train/calibration overlap in `split_reference` (the splits are disjoint).

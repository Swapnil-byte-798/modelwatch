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

# Drop HouseTours; decide Stage 3 on Ego4D alone

ADR-0002 accepted training on the union of Ego4D and HouseTours because each corpus is strong where
the other is weak. Access was the one prerequisite ADR-0002 did not yet have in hand for HouseTours.
It still does not. Stage 3 now proceeds on Ego4D alone.

## Status

accepted

## Context

Issue #20's acquisition work resolved all 644 EgoEnv Ego4D spans through 406 parent `video_540ss`
sources — zero unresolved identifiers. The Ego4D secret verified end to end with a real clip
download. HouseTours acquisition did not clear the same bar: the no-download probe requested 3
public source videos (a target of 2 plus 1 surplus) and all 3 returned YouTube's "Sign in to confirm
you're not a bot" challenge. Zero of the requested HouseTours sources succeeded. No HouseTours source
video, frame, or thumbnail was downloaded.

The spec's own acceptance criteria for issue #20 require HouseTours failures to "remain visible" and
"not silently reduce the requested sample" rather than be quietly worked around with a cookie hack or
a smaller surplus. Getting a working HouseTours source route (an authenticated download identity, or
an alternative public mirror) is a credentials/access decision, not an engineering one, and it is not
resolved.

## Decision

Stage 3 trains and reports **Run A only**: train and evaluate on Ego4D. Run B (held-out HouseTours)
and Run C (HouseTours-only training) do not run, because both need HouseTours source video that does
not exist locally or on the Modal volume. The frozen corpus manifest keeps the HouseTours rows from
the published EgoEnv annotations (their labels cost nothing and stay useful if access is solved
later), but every HouseTours clip resolves to `unresolved` or `unavailable` and none is selected into
any split.

Ego4D's held-out group is split into two **disjoint** groups, both drawn uniformly at random from the
same seeded generator: one fixes the decision-policy operating point (threshold, hysteresis), the
other gives the final threshold-transfer measurement. This is not new scope — issue #22's evaluation
protocol already required that "a group used to fix the operating point cannot provide the final
transfer measurement" — but it was not previously represented in the manifest schema. `corpus.py`'s
`build_frozen_manifest` now accepts an optional `test_counts` mapping alongside the existing
`heldout_counts`; omitting it reproduces the prior two-way train/heldout behavior.

## Consequences

- **The domain-shift measurement (run C) is lost for now.** There is no way to measure whether the
  union would have helped or hurt, because there is no HouseTours-trained head to compare against.
  Run A's absolute numbers are what decide the kill criterion regardless.
- **The reviewed hard-negative gap Ego4D never had gets worse, not better.** ADR-0002 already noted
  that only the archived 6-clip HouseTours subset had human-reviewed hard negatives. With HouseTours
  out entirely, every negative in the reported result comes from automatic room-interval derivation.
  This sharpens the existing open risk; it does not create a new one.
- **The environment-count generalization argument that motivated ADR-0002 is unresolved, not
  refuted.** Ego4D still tops out at 406 environments. A HouseTours access route (an approved
  authenticated identity, or a different public source) can restore runs B and C later without
  reprocessing Ego4D; nothing here is a scientific reversal of ADR-0002, only a scoping of what
  ships first.
- **The full Ego4D-only download is large: 117.456 GB** across the 406 resolved `video_540ss`
  sources, per the official CLI's own confirmation-prompt estimate (2026-08-30, `ego4d==1.7.3`,
  `v2_1`). That number is recorded and no transfer has started; the CLI was run without `-y` and
  declined its own prompt, matching issue #20's acceptance criterion.
- **This changes what "the corpus" means for the 20–30 minute encode-and-train budget.** Dropping
  HouseTours removes its full-clip encoding cost (HouseTours had no boundary-window shortcut — its
  clips are short enough that a boundary window covers almost the whole clip). What is left is
  Ego4D's boundary-window strategy for training clips and full-clip encoding for both held-out
  groups. Hitting the budget is a parallelism question, not a scope question: the corpus stays full:
  it is answered with enough concurrent Modal GPU containers, not with a smaller Ego4D sample.

## Considered options

**Retry the HouseTours probe with more surplus sources.** Rejected for now. The failure mode
(YouTube's bot challenge) does not improve with a larger candidate pool; it needs a different
credential or access route, which is outside this decision's authority to obtain.

**Quietly reduce the HouseTours request instead of recording the failure.** Rejected. This is exactly
what issue #20's acceptance criteria forbid: HouseTours failures must stay visible, not shrink the
sample silently.

**Block Stage 3 entirely until HouseTours access is solved.** Rejected. Ego4D alone still lets Run A
— the run that decides the kill criterion — proceed on schedule. Runs B and C are additive
generalization checks, not gates on Run A.

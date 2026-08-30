# Detect transitions with a supervised causal head, not an unsupervised score

An archived effort measured unsupervised change detection on frozen video-encoder world states. It
found a change of appearance and not a change of place: pooled area under the curve 0.622, and 0.592
for indoor-to-indoor boundaries, which touches chance. Stage 3 therefore trains a small supervised
causal head on labelled transitions instead of scoring novelty.

## Status

proposed

## Considered options

**Another unsupervised score.** Rejected. Fourteen configurations were tried — kNN cosine, diagonal
Mahalanobis, self-normalised ratio and robust z, `ruptures` KernelCPD, `skchange` MovingWindow and
PELT, and five gapped contrasts. Every one lands on the same floor of 0.49 to 0.61 for
indoor-to-indoor. An oracle that was given the true number of transitions scored 0.108 recall
against a 0.127 random baseline.

**Online change-point statistics** (CUSUM, Page-Hinkley, Bayesian online change-point detection, PCA
then detect). Rejected without further testing. These sit in the same family as the fourteen
configurations above. Re-testing them would rediscover a known result.

**Optical flow as a camera-motion gate.** Rejected. Four separate tests refuted it.

## Why the supervised head is the untried thing

A straddle contrast — a clean window before a known boundary against a clean window after it —
separates indoor-to-indoor at 0.76 to 0.79. The class information is present in the world states.
Every refuted method threw away the temporal structure that carries it.

The same record shows the causal form of that contrast at only 0.599. We hold that this gap is
alignment and not future information. A detector at time `t` holds every world state up to `t`, so
a detector allowed to answer late holds both windows. What the acausal contrast additionally has is
knowledge of where the boundary is. A trained temporal model can learn that alignment; a fixed
contrast cannot. This is why the head is recurrent and why a detection delay budget exists.

## Consequences

- Stage 3 needs labelled transitions. This makes the corpus a load-bearing decision. See ADR-0002.
- The archived numbers came from a different corpus. They give the shape of the problem, not a
  target. The 25 percent recall floor is therefore a judgement, not a derived value.
- Training a model contradicts the Stage 0/1 specification, which excludes it. That exclusion is
  redirected rather than deleted, so the reason it existed stays on the record. No model that
  produces a scene card is trained or fine-tuned.

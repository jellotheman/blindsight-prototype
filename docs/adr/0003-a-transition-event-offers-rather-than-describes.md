# A transition event offers, and never describes

When the detector fires, the client sounds one short earcon that means the surroundings changed. It
does not speak a description and it does not start a capture. The user decides whether to trigger
one.

## Status

proposed

## Why

**It is what the product already promises.** `CONTEXT.md` makes silence a functional state and makes
consent the rule: the system decides when something is worth offering, and the user decides whether
to hear it. An unrequested description takes that decision away.

**An automatic description would be poor even when the detector is right.** The user did not look
around. The camera points wherever the user happens to face, so the resulting scene card describes
whatever was in front of the lens rather than what the user chose to look at. Stage 0 exists
precisely because a description grounded in a deliberate captured view is better than one that is
not.

**It changes what accuracy the detector needs.** The cost of a false positive falls by roughly two
orders of magnitude — a wrong earcon spends a second of attention, a wrong description spends
thirty, plus provider latency and cost. A detector at 0.7 is shippable under this design and would
not be under the other.

## Considered options

**Speak a full scene card automatically.** Rejected for the three reasons above.

**Stay silent and only pre-warm a capture** so a later user-triggered one returns faster. Rejected:
zero attention cost, but it delivers none of the value the setting exists for. The user still has to
know the place changed.

## Consequences

- The earcon must be distinct from the four sounds Stage 0 defines: ready, captured, settled, and
  the failure buzz.
- The setting is off by default. The user turns it on.
- `Capture` keeps its existing definition: user-triggered, always. Stage 3 adds no automatic path
  into the capture flow, so the Stage 0/1 HTTP interface needs no change.
- The false-trigger budget of one per ten minutes is set against the cost of an earcon. If a later
  stage ever makes the trigger speak, that budget must be re-derived, not inherited.

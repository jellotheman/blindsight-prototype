# Ego4D room annotations decide the Stage 3 result

The archived effort measured transition detection on HouseTours, a set of real-estate walkthrough
videos from a video website. Stage 3 instead commits to the Ego4D room-prediction annotations, which
cover head-worn egocentric video and therefore match the BlindSight camera position. HouseTours
stays available as cheap, licence-free footage for shaking out a pipeline, but it never decides a
result.

## Status

proposed

## Why

Both annotation sets ship in the same EgoEnv archive, and Ego4D wins on nearly every axis.

| | Ego4D | HouseTours |
| --- | --- | --- |
| Usable room-change boundaries | 6478 | 6283 |
| Annotated duration | 176.9 h | 24.8 h |
| Boundaries per clip | 10.06 | 5.45 |
| Corrupt zero-length rows | 2 | 233 |
| Indoor-to-indoor share | 89.3 % | 76.9 % |
| Camera | head-worn | hand-held walkthrough |

The high indoor-to-indoor share reads as a defect and is not one. Indoor-to-indoor is the family
that collapsed to 0.592, and it is 54 of the 77 boundaries in the archived corpus. A corpus
concentrated there is a corpus concentrated on the real problem.

HouseTours video is steady, well lit and deliberately framed. A blind or low-vision user wears a
camera that is none of those things. A result measured on HouseTours would not transfer.

## Why a third-party annotation is unavoidable

Ego4D itself ships nothing that marks a change of place, so the EgoEnv labels are not a convenience
— they are the only option. `scenarios` is a flat array of strings on the video with no timestamps,
and its values are activities rather than places. `physical_setting_name` holds one value for an
entire video and is null except where a 3D scan exists. Moments annotate activities; NLQ is free
text; Hands & Objects, VQ and AV carry no location semantics at all. Narration summaries are the
closest native signal, and they are prose, so deriving transitions from them would be new work with
no ground truth to check it against.

The Ego4D licence explicitly permits this use. Its granted purpose enumerates software that detects
or understands a *place*, and permits commercial product development. Derived models and techniques
remain the end user's property. Redistribution of the data does not — no frame or clip may appear in
any shipped product, which constrains demonstrations but not the artifacts this stage produces.

## Consequences

- **Every archived number stops being a target.** The 0.622 pooled figure, the 0.592 floor and the
  0.76-to-0.79 ceiling are all HouseTours measurements. They describe the shape of the problem on
  different footage. The true ceiling for Ego4D is unknown.
- **The reviewed hard negatives are lost.** HouseTours had 84 ranges that a person reviewed. Ego4D
  has none, so negatives are derived automatically. Automatic negatives may be easier than real
  ones, and the measured false-trigger rate may therefore be optimistic.
- **Access becomes a prerequisite.** Ego4D needs a signed licence and AWS credentials, and its video
  is not on disk. A Modal secret `blindsight-ego4d-aws` exists but has never been used. Verifying it
  is the first acceptance criterion.
- **Encoding strategy changes.** Ego4D clips average about 16.5 minutes with roughly 10 boundaries,
  so boundaries sit about 100 seconds apart. Encoding a window around each boundary is worth roughly
  an order of magnitude. This was not worth doing on HouseTours, whose clips average 96 seconds.
- One new label appears, `recreation_room (billiards room / play room)`. The label-to-zone map needs
  one addition; the rest of the vocabulary is shared.

## Note on the target

The annotation records that the room changed. The product needs to know that a fresh description is
warranted. These are not the same thing, and no room-boundary corpus can close that gap. Stage 3
accepts the proxy deliberately and states it as an open risk rather than treating a good score as
proof of the product claim.

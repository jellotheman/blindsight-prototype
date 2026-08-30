# Train on both room-annotation corpora; Ego4D decides the result

The EgoEnv archive ships room annotations for two corpora, and each is strong exactly where the
other is weak. Stage 3 trains on the union. It reports the held-out **Ego4D** result as the one that
decides, because Ego4D is head-worn video and matches the BlindSight camera position. HouseTours is
not a fallback or a smoke test. It supplies the environment diversity that Ego4D lacks.

## Status

proposed

## The trade

| | Ego4D | HouseTours |
| --- | --- | --- |
| Usable room-change boundaries | 6478 | 6283 |
| Annotated duration | **176.9 h** | 24.8 h |
| Boundaries per clip | **10.06** | 5.45 |
| Corrupt zero-length rows | **2** | 233 |
| Indoor-to-indoor share | **89.3 %** | 76.9 % |
| Distinct source videos | 406 | **894** |
| Camera | **head-worn** | hand-held walkthrough |

Ego4D wins on hours, density, label quality and domain. HouseTours wins on the one axis that has
already broken this project: **count of distinct environments**.

That distinction is easy to miss. One HouseTours video is one house, so 894 videos are about 894
environments. One Ego4D video is one recording session from one camera wearer. A wearer can record
many hours inside a single home. Ego4D's environment count is therefore *at most* 406, and plausibly
far lower. Total hours are not the same thing as distinct places.

The archived effort failed across environments, not across time: 0.962 on one clip, 0.622 on six
unseen buildings. Environment count is therefore the axis that governs generalization. The union
gives roughly 12,700 boundaries across roughly 1,300 environments. Neither corpus gives both.

The high Ego4D indoor-to-indoor share reads as a defect and is not one. Indoor-to-indoor is the
family that collapsed to 0.592, and it was 54 of the 77 boundaries in the archived corpus. A corpus
concentrated there is concentrated on the real problem.

## Why a third-party annotation is unavoidable

Ego4D itself ships nothing that marks a change of place, so the EgoEnv labels are not a convenience
— they are the only option. `scenarios` is a flat array of strings on the video with no timestamps,
and its values are activities rather than places. `physical_setting_name` holds one value for an
entire video and is null except where a 3D scan exists. Moments annotate activities; NLQ is free
text; Hands & Objects, VQ and AV carry no location semantics at all. Narration summaries are the
closest native signal, but they are prose. Deriving transitions from them would be new work, with no
ground truth to check it against.

The Ego4D licence explicitly permits this use. Its granted purpose enumerates software that detects
or understands a *place*, and permits commercial product development. Derived models and techniques
remain the end user's property. Redistribution of the data does not. No frame or clip may appear in
any shipped product. This constrains demonstrations, but not the artifacts this stage produces.

## Considered options

**Ego4D alone.** Rejected. Right domain and best labels, but at most 406 environments, which is thin
for the failure mode this whole stage exists to escape.

**HouseTours alone.** Rejected. Most environments, no licence, and directly comparable with every
archived number — but hand-held property-tour footage is steady, well lit and deliberately framed. A
blind or low-vision user wears a camera that is none of those. A good result would not transfer.

**Train on HouseTours, evaluate on Ego4D.** Not rejected — kept as run C. It measures the domain
shift directly rather than assuming it, and it reveals whether the union helped or only added noise.

## Consequences

- **Three runs, not one.** Run A trains on the union and evaluates on held-out Ego4D; it decides
  against the kill criterion. Run B evaluates on held-out HouseTours as a second generalization
  check. Run C trains on HouseTours alone and evaluates on Ego4D to size the domain shift.
- **A new failure mode appears.** Two capture styles in one train set means the head could learn to
  separate the *corpora* rather than the *classes*. A run A result worse than run C is the signal
  that the union hurt.
- **Every archived number stops being a target.** The 0.622 pooled figure, the 0.592 floor and the
  0.76-to-0.79 ceiling are HouseTours measurements. They describe the shape of the problem. The
  ceiling for Ego4D, and for the union, is unknown.
- **The reviewed hard negatives are lost.** HouseTours had 84 ranges a person reviewed. Neither the
  full HouseTours set nor Ego4D has them, so negatives are derived automatically. Automatic
  negatives may be easier than real ones, and the false-trigger rate may be optimistic.
- **Access is no longer a prerequisite.** The Ego4D licence is signed and the credentials exist as
  the Modal secret `blindsight-ego4d-aws`. It has never been used, so it still needs one verifying
  download, and it expires around 2026-09-13.
- **The corpus builder handles two identifier schemes.** Ego4D rows carry `video_uid` and
  `clip_uid`; HouseTours rows carry a `clip_uid` encoding a video id and a 2 fps frame range.
- **The label map takes the union.** Nineteen labels are shared; Ego4D adds
  `recreation_room (billiards room / play room)`, HouseTours adds two more.

## Note on the target

The annotation records that the room changed. The product needs to know that a fresh description is
warranted. These are not the same thing, and no room-boundary corpus can close that gap. Stage 3
accepts the proxy deliberately and states it as an open risk rather than treating a good score as
proof of the product claim.

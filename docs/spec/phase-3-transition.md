# BlindSight Stage 3 transition-detector specification

**Status:** Specification only. No implementation exists.
**Primary artifact:** A trained causal transition detector and its streaming inference path
**Build order:** After Stage 0 and Stage 1. This document does not change the Stage 0/1 HTTP interface.

This document uses ASD-STE100 Simplified Technical English. Sentences are short. Each term has one
meaning. The document does not use a synonym for a term that this specification defines.

## Problem statement

A BlindSight user must trigger each capture. The user must know that the place changed before the
user can ask about it. A user who enters a new room does not always know that the room is new.

The product must tell the user that the surroundings changed. The product must do this without a
description that the user did not ask for. This is Stage 3.

The system must find the change while the video continues. The system cannot look at future frames.
A detector that looks at future frames does not operate in a live stream.

## Solution

The system computes a world state from the video at a fixed rate. The system sends each world state
to a small trained detector. The detector returns the probability of a transition. A decision policy
converts the probability into a transition event. The client sounds an earcon after a transition
event. The user then decides whether to trigger a capture.

The detector is small. The detector runs in a few microseconds. The encoder that produces the world
state is large. The encoder controls the compute cost of this stage.

## What the archived effort established

An archived effort measured unsupervised change detection on world states. The record is in
`blind-sight/docs/direction/prior-effort-findings.md`. This specification accepts these
measurements. Do not repeat them.

1. Unsupervised detection found a change of appearance. It did not find a change of place.
2. One clip gave an area under the curve of 0.962. Six unseen buildings gave 0.622.
3. Indoor-to-indoor boundaries gave 0.592. This value touches chance.
4. Fourteen detector configurations gave the same result. Calibration did not correct the failure.
5. An oracle that knew the true number of transitions gave 0.108 recall against a 0.127 baseline.
6. A contrast between a clean window before a boundary and a clean window after a boundary gave
   0.76 to 0.79. The class information is present in the world states.
7. Movement inside a room displaces the world state as much as movement between rooms.
8. Optical flow did not separate camera motion from a change of place. Four tests refuted it.

Item 6 is the reason to build a trained detector. Items 1 to 5 are the reason not to build another
unsupervised score.

The archived measurements used only 6 HouseTours clips and 77 boundaries. This specification uses
two corpora and about 12700 boundaries, and it decides on Ego4D. The values above therefore give the
shape of the problem. They do not give a target for this work.

## The causal problem

The archived record states a contrast value of 0.76 to 0.79 with a known boundary position. It
states 0.599 for the causal form of the same contrast. This specification holds that the difference
is alignment, and not future information.

A detector at time `t` holds every world state up to time `t`. A detector that answers 2 seconds
late therefore holds the window before the boundary and the window after the boundary. The acausal
contrast also knows where the boundary is. The causal detector must find the boundary while it
examines every time step.

A trained temporal model can learn this alignment. A fixed contrast cannot. This is the reason for a
recurrent detector and for a detection delay budget.

## Vocabulary

`CONTEXT.md` holds the full glossary. This specification adds four terms.

- **World state:** the 1024-dimension vector that the encoder computes from one trailing window of
  video frames.
- **Transition:** a change of place that is large enough to justify a fresh understanding of the
  surroundings.
- **Transition event:** the output of the decision policy. It reports that a transition occurred.
- **Proactive description:** an optional setting. When the setting is active, the system computes
  world states continuously and looks for transitions.

## Corpus

The system trains on two corpora and reports one deciding result. The EgoEnv authors published room
annotations for both. The file `egoenv-annotations.zip` holds both label sets. The annotations are on
disk. No video file is on disk.

| Property                      | Ego4D   | HouseTours |
| ----------------------------- | ------- | ---------- |
| Room-visit intervals          | 8681    | 7608       |
| Distinct clips                | 644     | 1152       |
| Distinct source videos        | 406     | 894        |
| Usable room-change boundaries | 6478    | 6283       |
| Annotated duration            | 176.9 h | 24.8 h     |
| Mean boundaries for each clip | 10.06   | 5.45       |
| Zero-length interval rows     | 2       | 233        |
| Indoor-to-indoor share        | 89.3 %  | 76.9 %     |

The Ego4D boundary families are as follows. Indoor-to-indoor is 5784 boundaries. Threshold-cross is
468 boundaries. Indoor-to-outdoor is 175 boundaries. Outdoor-to-outdoor is 51 boundaries.

The indoor-to-indoor share is high. This is correct for this specification. Indoor-to-indoor is the
family that the archived effort failed to detect.

### Why the system uses both

Each corpus is strong where the other is weak.

Ego4D video is head-worn video. This matches the BlindSight camera position. Ego4D also holds seven
times the annotated duration, twice the boundary density, and almost no corrupt rows.

HouseTours holds more distinct environments. One HouseTours video shows one house, so 894 videos
give about 894 environments. One Ego4D video is one recording session from one camera wearer, and
one wearer can record many hours in one home. The count of Ego4D environments is therefore not more
than 406 and is probably much less.

The archived effort failed across environments and not across time. A detector reached 0.962 on one
clip and 0.622 on six unseen buildings. The count of environments is therefore the axis that decides
generalization. HouseTours supplies that count. Ego4D supplies the domain.

The union holds about 12700 boundaries across about 1300 environments. Neither corpus gives both.

### Which result decides

**Held-out Ego4D clips give the result that decides.** Ego4D matches the deployment domain, so a
number from Ego4D is a number about the product.

**Held-out HouseTours clips give a second generalization check.** The implementer reports it
separately. It never replaces the Ego4D result.

The implementer must also report the Ego4D held-out result for a head that trained on HouseTours
alone. This measures the domain shift directly. It shows whether the extra environments help or
whether they only add noise.

### Label sources

Both label sets use the same schema and almost the same vocabulary. Ego4D holds 20 labels and
HouseTours holds 21. Nineteen labels are shared. Ego4D adds
`recreation_room (billiards room / play room)`. The label-to-zone map must hold the union of both.

Ego4D rows carry a `video_uid` and a `clip_uid`. HouseTours rows carry only a `clip_uid`, which
holds a video identifier and a frame range at 2 frames per second. The corpus builder must handle
both forms.

### Access

No video file is on disk. Each corpus needs a different source.

**HouseTours needs no licence.** The video comes from a public video website. The `clip_uid` holds
the video identifier and a frame range at 2 frames per second. A download tool such as `yt-dlp`
fetches it. Some videos are dead, private, or blocked by region. The implementer must request more
clips than the target count and must record which identifiers succeeded.

**Ego4D needs a signed licence.** The licence is already signed and the credentials already exist as
the Modal secret `blindsight-ego4d-aws`. The secret was created on 2026-08-30. The approval wait of
about 48 hours is therefore not on the critical path.

**Ego4D credentials expire after 14 days.** The current secret therefore expires at about
2026-09-13. Re-registration at `https://ego4d.dev/request/ego4d` returns new credentials
immediately. A `403 Forbidden` result on `HeadObject` almost always means that the credentials
expired.

The secret has never been used. Provisioned is not the same as working. The implementer must
download one clip end to end before the implementer plans a large download.

Install the download tool with `pip install ego4d`. The documentation names no version. The
implementer must pin one.

Ego4D publishes no annotation for a change of place. The `scenarios` field is one flat array of
strings on the video. It carries no timestamps, so it cannot express a transition. The values are
activities and not places. The `physical_setting_name` field holds one value for a whole video and
is null for most videos. Narration summaries give a coarse segmentation in prose, but they are not
labels. The third-party EgoEnv annotations are therefore necessary. See ADR-0002.

### Download

A clip is a separate file on S3 with its own `s3_path` and its own metadata. A clip is not only a
time range inside a full-scale video. The implementer can therefore download single clips.

```
ego4d --output_directory=<dir> --datasets clips --video_uid_file <file>
```

Four rules apply.

- The `--video_uids` filter works only for the `full_scale`, `clips`, `components/videos` and
  `video_540ss` datasets. The tool drops the filter silently for other datasets.
- The tool prints the exact total download size at the confirmation prompt. The implementer must run
  the command without `-y` and read that number before the implementer approves a large download.
- The tool defaults to version `v2_1` in code and to `v2` in its documentation. Version 2 removed
  some videos, so an older identifier does not always resolve. The implementer must first download
  the metadata alone, which is free, and compare the EgoEnv identifiers against `ego4d.json`.
- The `clips` dataset holds only the clips that a benchmark task exported. An EgoEnv `clip_uid` may
  have no file. The implementer must then cut the span from the parent full-scale video.

A `clips_540ss` variant may exist. The Ego4D pages name it but the tool documentation does not list
it. The implementer must run `--list-datasets` after the credentials work. A 540-pixel variant is
the cheapest source that this encoder can use, because preprocessing crops to 384 pixels.

Ego4D video runs at 30 frames per second. This confirms the `short` window configuration below.

### Label layout in time

The system derives a boundary from two adjacent room-visit intervals with different labels. The
system sorts the intervals of a clip by start time before it makes pairs. Some clips list intervals
out of time order.

The system assigns a label to every time step at the emission rate.

- The label is positive for 4 seconds after a boundary. This matches the detection delay budget.
- The label is `ignore` for a guard band on each side of the positive range. A trailing window that
  crosses a boundary holds frames from both places. The truth in this range is ambiguous.
- The label is negative everywhere else.

The system does not sample discrete negative events. The deployed detector scores every time step.
The measured false-trigger rate must therefore come from every time step.

Neither corpus has a reviewed hard-negative set at this scale. An archived effort reviewed 84
hard-negative ranges by hand, but those ranges cover only 6 HouseTours clips. Every other negative
comes from the room intervals alone. This is a known gap. See Open risks.

## World state extraction

The encoder is V-JEPA 2.1, distilled, ViT-L/16 at 384 pixels. The checkpoint file is
`vjepa2_1_vitl_dist_vitG_384.pt`. The licence is MIT. The source is the `facebookresearch/vjepa2`
repository at commit `204698b45b3712590f06245fbfba32d3be539812`. The implementer must pin both.

The encoder takes 64 frames at 384 by 384 pixels. Preprocessing resizes the short side to 438
pixels. Preprocessing then takes a centre crop of 384 pixels. Preprocessing applies the ImageNet
mean and standard deviation. The encoder runs under bfloat16 autocast. The encoder output is the
mean over the patch tokens. The result is a 1024-dimension float32 vector.

The window is **trailing**. A world state at time `t` uses frames at or before time `t`. The window
must never hold a frame after time `t`. A centred window leaks future frames. A detector that trains
on centred windows reports a value that does not survive deployment.

The emission rate is 1 Hz. This gives four scoring opportunities inside the 4-second delay budget. A
persistence rule of two consecutive high scores therefore fires with a margin.

### Window configurations

The frame sample rate sets the length of video that one window covers. This specification treats the
window length as a controlled factor. No earlier effort varied it.

| Name    | Frame sample rate | Video length per window |
| ------- | ----------------- | ----------------------- |
| `short` | 30 fps            | 2.13 s                  |
| `long`  | 4 fps             | 16 s                    |

The implementer must encode both configurations. The implementer must report both results. A 16-
second window holds 12 seconds of pre-boundary video at 4 seconds after a boundary. The `long`
configuration can therefore give a good area under the curve and still fail the delay budget.

### Compute cost

One window costs about 0.30 GPU-seconds on an A100-40GB. This value already includes bfloat16,
FlashAttention, and uint8 frame transfer. There is no further large saving in the encoder.

Full-corpus encoding costs about 53 GPU-hours for one configuration. This is too expensive.

The two corpora need different strategies, because their clips differ in length.

**Ego4D: encode a window around each boundary.** The mean clip is about 16.5 minutes and holds about
10 boundaries, so boundaries sit about 100 seconds apart. A window around each boundary costs about
one order of magnitude less than the full clip.

**HouseTours: encode the full clip.** The mean clip is about 96 seconds and holds about 5
boundaries, so boundaries sit about 18 seconds apart. A window around each boundary covers almost
the whole clip. There is nothing to remove.

**Held-out clips: encode the full clip, always.** The false-trigger rate needs continuous video. A
set of boundary windows does not hold enough ordinary video.

Download cost and encode cost are different. The tool downloads a whole clip file. The system then
encodes only the boundary windows inside it. A reduction in encode cost does not reduce the
download.

**Do not assume that a clip file is small.** Ego4D encodes clips at CRF 18 and full-scale videos at
CRF 41. CRF 18 is almost visually lossless. A clip can therefore hold more bytes for each second
than the same span of the parent video. The published figures give about 1.4 GB for each hour of
full-scale video. Ego4D publishes no figure for the `clips` dataset. The implementer must read the
real total from the confirmation prompt before the implementer commits to a clip count.

The implementer must use parallel Modal containers. A faster single GPU does not solve this. Sixteen
containers reduce a 2.4 GPU-hour job to about 9 minutes.

## Features

The feature vector holds 3076 values for each time step.

| Part          | Size | Content                                        |
| ------------- | ---- | ---------------------------------------------- |
| Normalized    | 1024 | the world state after normalization             |
| Delta         | 1024 | the world state minus the previous world state |
| Residual      | 1024 | the world state minus the previous EMA          |
| Scalars       | 4    | see below                                       |

EMA is the exponential moving average of the world states. The smoothing factor is 0.1.

The four scalars are as follows. The first is the cosine between the world state and the previous
world state. The second is the cosine between the world state and the previous EMA. The third is the
norm of the delta. The fourth is the norm of the residual.

Every part uses only current and past values. The EMA update must run one step at a time. An
implementation that computes the EMA over a full array is correct offline. It is not the streaming
form. The streaming form and the offline form must give the same result.

## Detector

The implementer must build two detectors and compare them.

**Logistic head.** A standardized linear model over the 3076 features. It returns one logit.

**Causal GRU head.** A linear layer from 1024 to 32. A tanh activation. A GRU with 32 hidden units.
A linear layer from 32 to 1. The history is 8 time steps.

The GRU head holds about 4000 parameters. Both heads run in microseconds. Neither head affects
system latency.

**Selection rule.** The implementer takes the GRU head only when two conditions are true. The GRU
head must beat the logistic head by 0.02 average precision or more. The GRU head must also win on
more than half of the held-out groups. This rule is fixed before training.

**Calibration.** The detector returns a raw logit and a calibrated probability. The calibration is a
slope and a bias over the logit.

## Decision policy

The policy converts a probability into a transition event. The policy holds a small state. The
policy updates in constant time.

| Parameter        | Value |
| ---------------- | ----- |
| Activate threshold | 0.8 |
| Release threshold  | 0.4 |
| Persistence        | 2 consecutive high scores |
| Cooldown           | 10 s  |

The two thresholds give hysteresis. The policy does not oscillate near one threshold.

### Why a threshold is acceptable here

A threshold destroyed the archived detector. This specification keeps a threshold. The difference is
what the threshold measures.

The system must make a sound or stay silent. That choice is binary. A probability alone therefore
cannot remove the decision. It can only move the decision to a later step.

The archived detector applied a threshold to an uncalibrated novelty value. That value has no fixed
scale, so a threshold from one building meant something different in another building. The archived
effort measured this. The usual correction made the result worse: self-normalisation gave a spread
of 1.87 times against 1.80 times for the raw value.

This detector applies a threshold to a calibrated probability. The calibration fits a slope and a
bias so that a value of 0.8 means about 80 percent. That meaning holds in a new environment, because
the calibration made it hold.

The policy is also more than a threshold. It needs two consecutive high scores. It uses a lower
value to release. It ignores a new event during the cooldown. One noisy time step cannot fire it.

The implementer must fix the threshold on held-out environments. The implementer must then measure
the false-trigger rate at that fixed threshold on other held-out environments. Threshold transfer is
therefore a measured property and not an assumption. See Acceptance criteria.

## Operating point

The implementer fixes these values before training. The implementer does not change them after the
implementer sees a result.

| Item                  | Value                    |
| --------------------- | ------------------------ |
| False-trigger budget  | 1 per 10 minutes or fewer |
| Detection delay budget | 4 seconds or less        |
| Recall floor          | 25 percent               |

The delay budget has a lower limit that no model can beat. A `short` window covers 2.13 seconds. The
world state at a boundary therefore already mixes both places. A detection below 2 seconds is not
available at any accuracy.

The archived effort used a budget of 1.53 false triggers per minute. That is one interruption every
40 seconds. That rate is not acceptable for a proactive setting.

**Kill criterion.** The implementer stops when recall at the false-trigger budget is below 25
percent on the held-out clips. The implementer then writes the result as a negative result. The
implementer does not tune the operating point to pass.

## Evaluation

The implementer holds out whole clips. The implementer never splits a clip across the train set and
the held-out set. Adjacent time steps in one clip are strongly correlated.

The implementer holds out a set from each corpus. The implementer selects boundary-dense clips for
the train set. The implementer selects held-out clips at random. A boundary-dense held-out set
raises the positive rate. That would make the false-trigger rate wrong.

### Three results

The implementer trains and reports three times. Each run uses both window configurations.

| Run | Train set | Held-out set | Purpose |
| --- | --------- | ------------ | ------- |
| A | Ego4D and HouseTours | Ego4D | The result that decides |
| B | Ego4D and HouseTours | HouseTours | A second generalization check |
| C | HouseTours only | Ego4D | The measured size of the domain shift |

Run A decides against the kill criterion. Run B shows whether the detector holds on a different
capture style. Run C shows whether the extra HouseTours environments help the Ego4D result or only
add noise. Run C also gives the value that a HouseTours-only effort could have reached.

### Reported values

The implementer reports the following for each run and for each window configuration.

- Recall at the false-trigger budget, for the held-out clips.
- Median detection delay for each detected transition.
- Recall for each boundary family. Indoor-to-indoor is the family that decides the result.
- Average precision for the logistic head and for the GRU head.
- The count of held-out groups that each head wins.
- The false-trigger rate at the fixed threshold, measured on the held-out clips. This value shows
  whether the threshold transfers.

## Artifacts

The work must produce these artifacts.

1. **Cached world states.** The causal world states for the corpus, on a Modal volume, in a compact
   binary format. This is the expensive artifact. Every later experiment then costs nothing.
2. **Trained heads.** The logistic head and the GRU head, with their calibration, in ONNX format.
   ONNX removes the PyTorch dependency at inference time.
3. **Streaming detector.** An incremental path. It accepts one world state. It returns a calibrated
   probability and a decision. It holds the EMA state, the feature history, and the policy state.
4. **Evaluation record.** The measured values above, and the corpus manifest. The manifest lists
   every clip identifier that the run used.

## Licence constraints

The Ego4D licence permits this work. The granted purpose names software that detects or understands
a place. It also permits commercial product development. The end user keeps the intellectual
property in every model and technique that the end user derives from the data. A world state is a
derived technique under that clause.

The licence forbids redistribution. No part of the data may appear in any product. This gives three
rules.

- The system may ship a trained head, a cached world state, and a measured result.
- The system must not ship a frame, a thumbnail, a clip, or any other part of the video. A
  demonstration must not show Ego4D video.
- The implementer must cite Ego4D, must keep copies internal, and must destroy every copy if the
  licence terminates.

The licence document is a draft and holds one agreement for each contributing university. The
implementer signs the agreements that cover the footage that the implementer downloads.

## Code layout

The detector code lives in a separate package inside this repository.

- The package is `blindsight/transition/`.
- The Stage 3 Modal application files are `modal_transition.py` (corpus acquisition) and
  `modal_transition_encode.py` (world-state encoding). Each file declares its own application name.
- The package declares its own image, its own volumes, and its own GPU functions.
- The ASGI application must not import the package. The package must not import the ASGI
  application.

This rule keeps the Stage 0/1 HTTP interface and its test seam provably unchanged. The OpenAPI
document does not change in this stage.

## Client behaviour

A transition event produces one short earcon. The earcon means that the surroundings changed. The
client does not speak a description. The client does not start a capture.

The user decides whether to trigger a capture. This follows the consent rule in `CONTEXT.md`. The
system decides when something is worth an offer. The user decides whether to hear it.

An automatic description is wrong for a second reason. The user did not look around. The camera
points wherever the user happens to face. A description from that video does not describe what the
user chose to look at.

The earcon must differ from the ready earcon, the captured earcon, the settled earcon, and the
failure buzz. Stage 0 defines those four sounds.

The proactive-description setting is off by default. The user turns it on.

## Out of scope

- Any change to the Stage 0/1 HTTP interface or the OpenAPI document.
- A hazard, safety, navigation, or mobility-aid claim. This stage makes no such claim.
- Persistent place memory across transitions. A transition event carries no history.
- Automatic capture. A transition event never starts a capture.
- Server-side speech. The client owns every sound.
- On-device or home-hub deployment of the encoder. See Open risks.
- The HouseTours corpus as the corpus that decides the result.
- Any statistical online change-point method. Items 1 to 5 above refuted this family.
- Optical flow as a gate for camera motion. Item 8 above refuted it.

## Acceptance criteria

- The Modal secret `blindsight-ego4d-aws` works. One Ego4D clip downloads end to end.
- The EgoEnv identifiers resolve against `ego4d.json`. The run manifest records the count that does
  not resolve, and the version that the implementer used.
- The implementer reads the download size from the confirmation prompt before a large download.
- The corpus builder reads both label sets and gives one table of boundaries with a corpus column.
- Every world state uses a trailing window. A test proves that no future frame enters a window.
- The streaming feature path and the offline feature path give the same values.
- Both window configurations produce cached world states for the same clip set.
- The held-out split contains whole clips only, and holds clips from one corpus only.
- The selection rule chooses between the two heads without a change to the rule.
- The implementer fixes the threshold on held-out environments, and then measures the false-trigger
  rate at that fixed threshold. The report states whether the threshold transferred.
- The evaluation reports all three runs. Run A gives the result that decides.
- The evaluation reports recall for each boundary family, and reports indoor-to-indoor separately.
- The streaming detector accepts one world state at a time and returns a decision.
- The ONNX export produces the same probability as the PyTorch model, within tolerance.
- A run manifest names every clip that the run used, and names the corpus of each clip.
- The Stage 0 test suite passes without change.

## Open risks

- **The encoder cost blocks local deployment.** The head is free. The encoder costs 0.30
  GPU-seconds for each window. At 1 Hz this is about 30 percent of a sustained A100. A home hub
  cannot supply this. A smaller head does not change this. This risk decides whether Stage 3 ever
  runs outside a datacentre.
- **The corpus has no reviewed hard negatives.** The archived effort had 84 ranges that a person
  reviewed. Ego4D has none. Automatic negatives may be easier than real negatives. The measured
  false-trigger rate may therefore be optimistic.
- **The target is a proxy.** The room-visit annotation reports that the room changed. The product
  needs to know that a fresh description is warranted. These are not the same. A good result on this
  corpus does not prove the product claim.
- **Ego4D video volume.** The annotations sit inside long videos. The download cost is unmeasured.
  Ego4D publishes no size for the `clips` dataset, and CRF 18 encoding can make a clip larger for
  each second than its parent video. The cost could be an order of magnitude above the estimate.
- **The credentials expire after 14 days.** The current secret dates from 2026-08-30 and therefore
  expires at about 2026-09-13. A download that starts late fails part way. Re-registration returns
  new credentials immediately, so this is a delay and not a block.
- **Two capture styles in one train set.** HouseTours video is steady and Ego4D video is not. A head
  that trains on the union can learn a feature that separates the two corpora instead of a feature
  that separates the two classes. Run C exists to detect this. A run A result that is worse than
  run C means that the union hurt.
- **Clip coverage is unknown.** The `clips` dataset holds only the clips that a benchmark exported.
  The number of EgoEnv identifiers with no clip file is unmeasured. Those spans need a cut from the
  parent full-scale video, which costs much more to download.
- **Version drift.** Ego4D version 2 removed some videos. EgoEnv published its annotations in
  December 2023. Some identifiers may no longer resolve.
- **The archived ceiling does not transfer.** The 0.76 to 0.79 value came from HouseTours. Ego4D is
  shakier and less well lit. The true ceiling for this corpus is unknown.
- **A continuous camera has a privacy cost.** The proactive setting encodes video without a user
  action. This specification does not solve that.

## Notes

- The branch `codex/transition-detector` in the private reference repository holds an earlier
  attempt at this pipeline. That attempt never ran. This specification keeps its design choices and
  discards its implementation.
- `housetours_selection.json` is redundant. The annotation CSV files hold the same intervals.
- HouseTours needs no licence, so it is also the fastest way to shake out the pipeline before an
  Ego4D download. This is a convenience and not the reason to use it. The reason is the environment
  count. See ADR-0002.
- The 6 clips and 77 boundaries of the archived effort are a subset of the HouseTours labels. The
  full HouseTours set holds 6283 boundaries across 1152 clips.

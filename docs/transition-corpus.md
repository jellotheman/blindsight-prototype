# Stage 3 transition corpus

The Stage 3 corpus builder lives in `blindsight.transition`. It is separate from the Stage 0/1
ASGI application and records only annotation and source identifiers in Git-facing artifacts. Source
media, credentials, downloaded manifests, and verification media stay on the private Modal data
volume.

## Frozen conventions

These values are declared before a frozen manifest is written:

- Ego4D CLI: `ego4d==1.7.3`
- Ego4D data version: `v2_1`
- Preferred parent source: `video_540ss`
- Positive proxy interval: four seconds after a boundary
- Guard band: one second on either side of that positive interval
- Held-out selection: uniform random sampling of complete resolved clips, using the recorded seed
- Training selection: remaining complete resolved clips, ordered by proxy-boundary count and then
  clip identifier

The builder reads all RoomPred CSV splits for Ego4D and HouseTours. It sorts each clip's non-zero
room intervals by start time. It creates a proxy boundary only for a changed label or room instance
between consecutive retained intervals. Each timestamp is then positive, ignored, or negative;
there is no discrete negative-event sampler.

HouseTours IDs have the form `<video-id>_<start-frame>_<end-frame>`. The final two fields are source
frame bounds at 2 fps, not seconds. The frozen manifest preserves those fields and marks its
expected extraction strategy.

## Acquisition checks

`modal_transition.py` attaches the `blindsight-ego4d-aws` secret only to the functions that require
it. It writes an ephemeral named AWS profile, invokes the official CLI, and never returns secret
values. The `preview_ego4d_download` operation deliberately omits `-y` and declines the CLI prompt;
it records the CLI's exact reported size before any large transfer.

The 2026-08-30 verification used one resolved `video_540ss` parent source. The CLI preview reported
0.3163 GB, then the private Volume retained a 339,630,462-byte file and an idempotent rerun confirmed
it. The official `v2_1` manifests resolve all 644 annotated Ego4D spans through 406 parent 540p
videos; no current direct clip file resolved. This is recorded as a parent-cut strategy, not silently
treated as a clip download.

## Count comparison

The official annotation archive has the specified 8,681 Ego4D and 7,608 HouseTours room intervals,
644 and 1,152 clips, 406 and 894 source videos, and 2 and 233 zero-length rows. The builder's current
ticket-defined rule derives 6,763 Ego4D and 6,077 HouseTours proxy boundaries. These differ from the
older specification totals of 6,478 and 6,283. The corpus report requires an explicit explanation
for every such difference; it must not be bypassed by changing the derived rows or silently dropping
source identifiers.

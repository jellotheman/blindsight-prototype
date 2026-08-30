# BlindSight Stage 0/1 specification

**Status:** Ready for implementation  
**Primary artifact:** Text-only HTTP interface  
**Build order:** Stage 0 first; Stage 1 is the explicit cut line

## Problem Statement

Blind and low-vision people can ask existing tools to describe a single camera image, but a still
often misses the relationships, activity, and visual impression available when someone deliberately
looks or points the camera across a place. A long exhaustive description consumes attention, while
a confident description can also imply complete or current awareness that the captured evidence
does not support.

BlindSight needs the smallest useful experience: the user triggers a short capture, receives a
concise spoken orientation grounded only in that captured view, and can ask for more detail without
hearing an inventory first. The result must be usable by multiple clients without coupling the
backend to browser speech, React Native, or one visual interface.

## Solution

Provide a front-end-agnostic HTTP interface that turns either an eight-second live capture or a
preloaded demonstration excerpt into a validated, revisable scene card. The interface is text-only.
Clients speak the overview and own all interaction sounds.

Stage 0 creates one fresh scene session per capture and returns a concise orientation. Stage 1 adds
conversational questions grounded in that same scene card. When the card lacks a requested detail,
the client asks permission before the backend checks the stored captured view again. A second miss
abstains.

The deployed backend and reference web client share one Modal application. The reference client has
no privileged path: it uses only the documented HTTP interface, exactly like a React Native client.

BlindSight is an environmental-understanding tool, not a navigation or mobility aid. It never makes
a safety claim.

## User Stories

1. As a blind or low-vision user, I want to trigger a capture without locating a small visual
   control, so that I can begin independently.
2. As a user, I want one tap anywhere on the primary client surface to start a capture, so that the
   initial interaction is simple.
3. As a user, I want an audible ready cue, so that I know the client accepted my action.
4. As a user, I want a short, fixed capture interval, so that I know how long evidence is being
   gathered.
5. As a user, I want to point or look around naturally during the capture, so that I choose what is
   described.
6. As a user, I do not want a required turn, pace, direction, or coverage target, so that normal
   selective use is not treated as failure.
7. As a user, I want a metronome during capture, so that I know recording remains active.
8. As a user, I want a distinct capture-complete cue, so that I know the evidence interval ended.
9. As a user, I want bounded processing cues, so that silence is distinguishable from a crash.
10. As a user, I want the first spoken result to be concise, so that it respects my attention.
11. As a user, I want the first result to identify the place type when evident, so that I can
    orient quickly.
12. As a user, I want occupancy and apparent activity included when observed, so that I understand
    who is present without identity claims.
13. As a user, I want dominant landmarks and relationships prioritized over inventory, so that the
    overview is useful rather than exhaustive.
14. As a user, I want object identities stated plainly, so that terseness does not replace useful
    names with vague warnings.
15. As a user, I want uncertainty attached to the claim it qualifies, so that I know exactly what
    may be wrong.
16. As a user, I want the system to abstain rather than invent, so that missing evidence does not
    become confident belief.
17. As a user, I want descriptions framed as observations from the captured view, so that a past
    recording is not presented as current reality.
18. As a user, I want directly observed colour, light, and materials when they fit, so that the
    result conveys visual impression as well as layout.
19. As a user, I want inferred style or atmosphere qualified as interpretation, so that inference
    is not presented as perception.
20. As a user, I want to ask follow-up questions, so that details remain available on demand.
21. As a user, I want follow-up questions to remember earlier questions within the scene session,
    so that references such as “it” work naturally.
22. As a user, I want quick questions answered from the scene card first, so that I avoid needless
    model latency and cost.
23. As a user, I want to be asked before the stored capture is checked again, so that I control the
    additional wait.
24. As a user, I want a plain abstention after a second miss, so that the interaction ends honestly.
25. As a user, I want a new capture to start a fresh scene session, so that observations from two
    captured views are not silently combined.
26. As a user, I want “done” to clear the active conversation, so that later questions do not use a
    stale scene card.
27. As a user, I want network and provider failures distinguished from successful completion, so
    that I do not wait indefinitely.
28. As a teammate building an Android client, I want an OpenAPI contract and executable examples,
    so that I can build before the backend is complete.
29. As a client developer, I want stable error codes and state transitions, so that recovery does
    not depend on parsing prose.
30. As a client developer, I want the interface to return text and structured data only, so that I
    can choose platform-native speech.
31. As a demonstrator, I want runtime-selectable preloaded excerpts, so that the experience remains
    demonstrable on a congested connection.
32. As a demonstrator, I want live and excerpt sources to produce the same scene-card shape, so that
    the listener hears one product rather than two demos.
33. As an operator, I want one shared API key, so that a public deployment cannot incur anonymous
    provider spend.
34. As an operator, I want asynchronous jobs stored outside container memory, so that polling works
    when requests land on different containers.
35. As an operator, I want usage recorded in provider units rather than capture counts, so that
    later streaming work does not require a cost-model rewrite.
36. As an evaluator, I want every run to retain the capture, validated card, timings, usage, and raw
    provider text, so that prompt or schema changes can be re-judged without repeating the room.
37. As a future implementer, I want capture identity, revisable cards, and asynchronous status now,
    so that later streaming can replace ingestion without replacing the public interface.

## Stage boundary

### Stage 0 — required

Stage 0 accepts an eight-second live capture or one preloaded excerpt and produces a valid scene
card. The client speaks the overview and then becomes silent. Every Stage 0 acceptance criterion is
independent of follow-up questions.

### Stage 1 — explicit cut line

Stage 1 adds scene-card-first questions, conversational context, permissioned captured-view checks,
and plain abstention. It is specified in full but built after all Stage 0 criteria pass.

## Product flow

### Stage 0 live capture

1. The client obtains camera permission through `getUserMedia` and keeps the `MediaStream` as the
   permission foundation.
2. One user action unlocks audio, starts a fresh scene session, and opens a live capture job.
3. The client says, “Look around at what you’d like described.” It does not prescribe movement.
4. The client records for eight seconds and uploads numbered media chunks as they become available.
5. Chunks may arrive out of order. The backend assembles them strictly by declared index, rejects
   gaps, and verifies that the resulting clip is decodable before invoking a provider.
6. The client marks the capture complete. The backend returns immediately and processes the job
   asynchronously.
7. The client polls until the capture succeeds or fails while running the wait ladder.
8. On success, the client speaks the scene-card overview and then the settled cue.

### Stage 0 demonstration excerpt

1. The client lists preloaded excerpts and selects one by stable identifier.
2. Starting the capture sends only the excerpt identifier. The clip already resides on the Modal
   volume.
3. The same provider orchestration, validator, capture resource, scene-card response, speech, and
   failure handling are used after source resolution.
4. The client must not present excerpt timing as live capture-end-to-first-word timing.

### Stage 1 follow-up

1. The client submits the question to the active scene session.
2. A text model receives the current scene card and this scene session’s conversation. It returns
   `answer` or `null` using only that evidence.
3. If it answers, the client speaks the answer.
4. If it returns `null`, the interface enters `needs_clip_consent`. The client offers to check the
   stored captured view and warns that this may take several seconds.
5. Only an explicit client request after user consent starts the video-model check.
6. If the video model answers, the client speaks it. If it returns `null`, the client says, “I
   couldn’t tell from the capture.”
7. “Done,” a new capture, or application shutdown ends the scene session. The client deletes it
   explicitly. The shared key carries no client identity, so the backend cannot infer which earlier
   scene session a new capture supersedes.

## Scene-card contract

The OpenAPI document is the normative JSON Schema. The semantic rules below are equally binding:

- `capture_id`, `scene_session_id`, `revision`, and `evidence` are backend-authored bookkeeping.
- A revision starts at `1`. Later evidence for the same capture may increase it; Stage 0 normally
  returns revision `1`.
- `overview` targets 30–45 words and has a hard maximum of 50 words.
- The overview prioritizes: place type; occupancy/activity; two or three dominant landmarks and
  relationships; open-space shape when material; then one short visual-impression clause if it
  does not displace orientation.
- `layout: null` means the provider could not determine layout. `layout: []` means it assessed the
  evidence and observed no layout items worth recording.
- `people: null` means presence could not be determined. `people: []` means no people were observed.
- Empty strings are invalid. Keys are never silently omitted.
- Each uncertainty is an object containing the affected `claim` and the qualifying `detail`.
- `uncertainties` is `null` when nothing is genuinely uncertain. An empty uncertainty array is
  invalid; ritual disclaimers are forbidden.
- Confidence labels are stored but never spoken as labels or numbers.
- `visual_character` contains only observed colours, lighting, and materials.
- The Stage 0/1 contract has no mandatory starting heading, clock position, turn measurement, or
  room-coverage field. Relationships are evidence-grounded natural language.
- Scene cards describe the captured view. They must not imply complete or current awareness.

## HTTP interface

All `/v1` routes require the shared `X-API-Key` header. The OpenAPI document defines the exact
requests, responses, status codes, schemas, and security scheme.

The interface exposes:

- excerpt listing and poster retrieval;
- creation of live or excerpt-backed captures;
- idempotent indexed chunk upload;
- capture completion and polling;
- scene-session question creation and polling;
- explicit captured-view checks after consent;
- scene-session deletion.

Capture and question processing are asynchronous resources. A successful submission returns `202`
or `201`; clients poll the resource URL and respect `Retry-After` when present. Asynchronous
provider failures appear as a stable failure object on the resource rather than changing the
already-returned submission response.

Deleting a scene session ends its conversation. Question operations on that scene session then
return `NOT_FOUND`, while the capture resource stays readable so a client can re-read the card it
already spoke. Retained evidence is unaffected.

## Implementation Decisions

- Build one backend module whose public interface is the documented HTTP contract. Both the web
  reference client and Android client are ordinary callers.
- Use a Python ASGI application deployed in the same Modal application as the reference client.
- Store capture jobs, scene sessions, question jobs, and conversation state in distributed Modal
  storage. Never rely on container memory across requests.
- Accept live media as numbered chunks and make repeated upload of identical chunk bytes
  idempotent. Reject the same index with different bytes.
- Repair a live capture assembled from streamed chunks through `ffmpeg` before validating
  decodability or invoking a provider. `MediaRecorder` writes WebM/Matroska (VP8/VP9) in streaming
  mode and never patches the segment Duration/seek metadata once recording stops; `ffprobe`
  accepts the result, but Reka's ingestion cannot decode it -- it parses the container and then
  yields zero frames. A native iOS client records QuickTime/MOV (`video/quicktime`), and container
  type alone does not rescue a capture: any codec other than H.264 (e.g. iOS HEVC) yields the same
  zero-frame failure. Transcode a live WebM -- or any non-H.264 live capture, including MOV -- to
  H.264 MP4 (the codec/container pair the preloaded excerpt path already hands Reka successfully);
  copy-remux an already-H.264-in-MP4 live capture to repair its streaming metadata without a lossy
  re-encode. Retained evidence keeps the repaired clip, since that is what a provider actually saw.
- Validate assembled media before provider spend. A corrupt or incomplete capture is a capture
  failure, not a model failure.
- Keep provider-specific ingestion behind internal adapters. The public interface exposes one
  normalized scene card and does not expose Reka/Gemini response shapes.
- Use Reka Chat through the OpenAI-compatible endpoint as the primary visual provider. Resolve the
  model from configuration, defaulting to the stable `reka-flash` alias.
- Make an assembled live clip or stored excerpt available to Reka through a short-lived,
  unguessable HTTPS media URL. The URL is an internal provider transport, not a public client route.
- Put the complete scene-card schema in the prompt, and additionally send it to Reka as an enforced
  `response_format: {"type": "json_schema", ...}` (undocumented but supported; see Provider and
  network facts). Send no assistant prefill: combining one with any `response_format` corrupts
  Reka's output. Validate the completed response with the sole scene-card validator.
- Allow two Reka attempts. When both attempts end in malformed or schema-invalid results *or* in
  transport or timeout failures, invoke the Gemini fallback once using its verified response-schema
  support. A transport or timeout failure consumes a Reka attempt; it never grants Reka more
  attempts or skips the fallback.
- Treat provider transport errors and timeouts separately from parse failures. Retry policy must be
  bounded and visible in retained evidence.
- Use Gemini as the measured fallback, not as a silent second opinion. Record which provider and
  attempt produced the accepted card.
- Keep Reka Vision and self-hosted open-weight VLMs out of the Stage 0/1 implementation.
- Store usage in provider-native units, including input/output tokens and any billed media units.
- Retain each capture, validated card, timings, usage, raw model response, selected provider, and
  failure record as evidence. Evidence is never read back into a different live scene session.
- Attach Modal secrets when declaring the remote function. Do not probe for a secret from inside a
  running container and conditionally re-register the function.
- Use one shared static API key in a Modal secret. Do not implement users, accounts, per-client keys,
  or OAuth in Stage 0/1. The reference client is an ordinary caller holding the key its user entered.
- Keep local development and deployment equivalent at the HTTP interface: Cloudflare quick tunnels
  expose the local server to a phone; the Modal web endpoint is the deployed path.
- Keep the reference client on the same deployment and forbid privileged in-process access.
- Do not implement server-side speech. Browser SpeechSynthesis and Android-native/expo speech are
  client choices.

## Provider and network facts

- Reka's documented stable Chat model alias is `reka-flash`, but as of 2026-08 that alias 404s
  ("Unknown chat model"). Deployments must resolve a live model id from configuration after
  listing `GET /v1/models`; the current deployment uses `reka-edge-2603` via Modal secrets.
- Reka Chat accepts short video by reachable `video_url` and is documented as best below 30
  seconds. The docs show `video_url` as a flat string, but the API requires the nested
  `{"video_url": {"url": ...}}` dictionary form; a string input returns a 400 validation error.
  On the live fleet only `reka-edge-2603`, `qwen3.8-flash`, and `qwen3.8-27b` accept video input;
  `reka-flash-3` is text-only despite its name.
- Reka Chat decodes H.264 MP4 but not browser WebM (VP8/VP9): a VP9-in-WebM `video_url` parses as
  a valid container yet yields zero decoded frames (`Expected 6 frames, got 0`). Live WebM
  captures must be transcoded to H.264 MP4 before they reach Reka.
- Reka Chat does not document JSON Schema response enforcement, but empirically accepts
  `response_format: {"type": "json_schema", ...}` and honors it strictly (3/3 valid cards against
  a public sample video, 2026-08-30). Under bare `json_object` mode the model echoes a schema
  embedded in the prompt back instead of answering, and combining any `response_format` with an
  assistant prefill corrupts the output. Prefix handling, parse, validate, retry, and the measured
  Gemini fallback therefore remain required defense.
- A live capture necessarily crosses the phone’s uplink once when chunks are sent to Modal.
  `video_url` prevents a second phone-to-provider transfer: after assembly, Reka fetches from
  Modal’s datacenter-facing URL. The preloaded excerpt path sends no video from the phone.
- Reka quality, validated-card rate, and latency for this exact Modal-hosted URL flow remain
  unmeasured. This is the only unmeasured provider risk accepted by this specification.

## Client requirements

Any client claiming Stage 0 support must implement the same interaction meanings:

- The primary surface is one full-page tap target with the instruction “Tap the center of the
  screen to record.”
- The capture instruction is “Look around at what you’d like described.”
- A distinct ready earcon confirms the trigger.
- A continuous metronome sounds during the eight-second capture.
- A distinct captured earcon marks the end of the evidence interval.
- Processing begins in silence; a quiet pulse starts at three seconds; “Still working.” is spoken
  once at eight seconds.
- A non-musical buzz identifies a network or terminal processing failure.
- A distinct settled earcon follows the spoken overview.
- The first spoken content is the scene-card overview. No card inventory is read automatically.
- Speech uses platform-native client TTS. The web and Android voices may differ.
- Stage 1 clients must speak the consent offer before requesting a captured-view check.
- Clients must never turn a null answer into a confident negative claim.
- The client asks its user for the shared API key once and keeps it in browser or device local
  storage. The key is never baked into a client build and never embedded in a page served by the
  deployment.
- Poster images are fetched with the `X-API-Key` header and rendered from an object URL. A bare
  image source cannot send the header.

The tap target and audio ladder are specified behavior, not verified blind interaction. Nothing has
been tried with the screen off or eyes closed. VoiceOver/TalkBack verification, local covered-lens
or blur detection, and complete screen-reader semantics remain out of scope.

## Error behavior

Every immediate HTTP error uses the shared error envelope with a stable `code`, human-readable
`message`, `retryable` flag, and optional structured `details`. Clients branch on `code`, never the
message.

Asynchronous failure codes include:

- `CAPTURE_INCOMPLETE`
- `CAPTURE_UNDECODABLE`
- `PROVIDER_TIMEOUT`
- `PROVIDER_UNAVAILABLE`
- `MODEL_OUTPUT_INVALID`
- `INTERNAL_ERROR`

No job may remain in `processing` indefinitely. Provider calls and the overall job both have bounded
deadlines. A terminal failure stops wait cues and produces the client failure buzz.

## Testing Decisions

The public HTTP interface is the single primary test seam. Tests use the real ASGI application,
serialization, authentication, route handling, state transitions, validator, and error mapping,
while substituting deterministic provider and distributed-store adapters.

Good tests assert externally visible behavior:

- response status, headers, and body validate against OpenAPI;
- authentication is required consistently;
- live and excerpt sources converge on the same scene-card response;
- chunks assemble by declared index even when requests arrive out of order;
- identical repeated chunks are idempotent and conflicting repeats fail;
- missing chunks and undecodable clips fail before provider invocation;
- capture completion returns before inference finishes;
- polling survives different application/container instances sharing one store;
- valid model output becomes one canonical scene card;
- two failed Reka attempts -- invalid output, transport, or timeout -- invoke Gemini exactly once;
- provider timeout, transport failure, and invalid output remain distinct;
- uncertainty null/array semantics and claim anchoring are enforced;
- a new capture creates a new scene session;
- Stage 1 card answers, consent states, captured-view answers, second misses, conversation context,
  and deletion are visible through HTTP only;
- every worked `curl` and `fetch` example remains valid against the OpenAPI document.

Prior art is the prototype’s whole-loop FastAPI TestClient test: start capture, deliver chunks,
finish asynchronously, poll, validate the card, and retain evidence. Its out-of-order chunk
regression is retained at the higher public interface seam.

Live Reka and Gemini calls are opt-in smoke tests requiring secrets. They record results but do not
run in the default acceptance suite. A live Reka smoke is required before claiming measured Reka
quality or latency.

The OpenAPI acceptance bar is independent: give only the OpenAPI document and examples to an
assistant with no repository access. The client it produces must authenticate, complete Stage 0,
poll correctly, render failures, and perform the Stage 1 consent flow without guessing undocumented
shapes.

## Acceptance criteria

### Stage 0

- A client can implement the full Stage 0 flow from OpenAPI and examples alone.
- All `/v1` operations reject a missing or incorrect API key with the documented error.
- A live capture accepts eight seconds of indexed chunks, detects gaps/corruption, processes
  asynchronously, and returns a valid scene card.
- A preloaded excerpt is runtime-selectable and returns the same scene-card schema through the same
  capture resource.
- Reka output is accepted only after canonical validation; failed Reka attempts -- invalid output,
  transport, or timeout -- fall through to Gemini exactly once.
- The accepted card obeys null/empty semantics, claim-specific uncertainty, visual-impression
  limits, and the 50-word overview maximum.
- Job state survives POST and GET requests landing on different containers.
- Every terminal path is bounded and produces a documented success or failure state.
- The reference web client uses only the public HTTP interface and implements the client audio
  ladder.
- Retained evidence is sufficient to re-run a prompt/schema judgment without returning to the
  original place.
- No Stage 0 criterion depends on Stage 1.

### Stage 1

- Questions are grounded in the active scene card and scene-session conversation first.
- A card miss returns `needs_clip_consent` without invoking the video provider.
- Only the explicit captured-view-check operation invokes the video provider.
- A captured-view answer and a second miss are distinguishable.
- Answers use capture-scoped language whenever freshness matters.
- Scene-session deletion prevents later access to that conversation, and every new capture starts a
  fresh scene session that never inherits earlier conversational context.

## Out of Scope

- Required or optional guided 360-degree room capture.
- Automatic place-transition detection. Specified in `phase-3-transition.md`.
- Relative depth or reliable spatial geometry.
- Continuous video streaming as an implemented capability. Continuous world-state extraction for
  transition detection is specified in `phase-3-transition.md`; continuous streaming as an ingestion
  path for scene cards remains out of scope everywhere.
- Proactive reminders. A Stage 3 transition event produces one earcon and never a description or a
  capture; see `phase-3-transition.md`.
- Persistent multi-room memory.
- Training or fine-tuning any model. Stage 3 trains a transition detector on frozen encoder output;
  see `phase-3-transition.md`. No model that produces a scene card is ever trained or fine-tuned.
- Safety, hazard detection, navigation, or mobility-aid behavior or claims.
- A native application implementation or app-store distribution.
- Server-side text-to-speech.
- Reka Vision managed indexing.
- Self-hosting Qwen or another open-weight VLM.
- An evaluation apparatus, direction-accuracy benchmark, or recruitment of BLV testers.
- VoiceOver/TalkBack verification beyond the single tap target.
- Local covered-lens, insufficient-light, or blur checks.
- Human-fallback implementation. A Be My Eyes-style handoff remains stated design intent.
- Stages 2–5.

## Further Notes

- The only real-room Stage 0 result is one dorm room, one Android phone, and one lighting condition.
  It took 8.6 seconds from capture end to first spoken word, including 6.9 seconds of Gemini model
  time. The result is evidence that the path works, not a reliability estimate.
- The same field capture produced 0.62 hours mean clock error against a 3.00-hour chance baseline
  when scored against the 91-degree arc actually recorded. Clock positions and the framing prompt
  are not part of this contract; framed and naive prompts were indistinguishable.
- The demonstration library contains 74 eight-second excerpts across eight videos. That footage
  shows someone walking rather than turning from a fixed facing, so it cannot produce a comparable
  direction-accuracy result.
- Nothing has been tried with the screen off or eyes closed.
- Human fallback is part of the honest product stance but is not built in Stage 0/1.
- Official provider references: [Reka models](https://docs.reka.ai/chat/models),
  [Reka multimodal Chat](https://docs.reka.ai/chat/chat-with-image-video-and-audio), and
  [Reka Chat API](https://docs.reka.ai/chat/api-reference/create).

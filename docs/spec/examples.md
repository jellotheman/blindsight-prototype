# BlindSight HTTP examples

These examples use placeholder values. Replace the base URL with the Modal deployment or local
Cloudflare URL and provide the shared key out of band.

```bash
export BLINDSIGHT_URL="https://example--blindsight-web.modal.run"
export BLINDSIGHT_API_KEY="replace-me"
```

Never commit the real key or embed it in a distributed client build.

## Stage 0: live capture with `curl`

### 1. Open the capture

```bash
curl --fail-with-body \
  -X POST "$BLINDSIGHT_URL/v1/captures" \
  -H "X-API-Key: $BLINDSIGHT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"source":{"type":"live","mime_type":"video/webm"}}'
```

Example response:

```json
{
  "capture_id": "cap_01J67J4JYG9X5Q20MZB6S8N7P3",
  "scene_session_id": "ses_01J67J4K0BT3WD5H1RV2N7Q6XM",
  "source": {"type": "live", "mime_type": "video/webm"},
  "status": "recording",
  "card": null,
  "failure": null,
  "created_at": "2026-08-29T11:30:00Z",
  "updated_at": "2026-08-29T11:30:00Z"
}
```

### 2. Upload numbered chunks

Chunk indices are zero-based. Requests may run concurrently and arrive in any order.

```bash
CAPTURE_ID="cap_01J67J4JYG9X5Q20MZB6S8N7P3"

curl --fail-with-body \
  -X PUT "$BLINDSIGHT_URL/v1/captures/$CAPTURE_ID/chunks/0" \
  -H "X-API-Key: $BLINDSIGHT_API_KEY" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @chunk-000.webm

curl --fail-with-body \
  -X PUT "$BLINDSIGHT_URL/v1/captures/$CAPTURE_ID/chunks/1" \
  -H "X-API-Key: $BLINDSIGHT_API_KEY" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @chunk-001.webm
```

Uploading byte-identical content to the same index again succeeds with `idempotent: true`. Different
bytes at an occupied index return `409 CHUNK_CONFLICT`.

### 3. Complete capture

```bash
curl --fail-with-body \
  -X POST "$BLINDSIGHT_URL/v1/captures/$CAPTURE_ID/complete" \
  -H "X-API-Key: $BLINDSIGHT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"chunk_count":8,"mime_type":"video/webm"}'
```

The response is `202` with `status: processing`. Missing indices produce a retryable
`409 CAPTURE_INCOMPLETE` with `details.missing_indices`.

### 4. Poll for the scene card

```bash
curl --fail-with-body \
  "$BLINDSIGHT_URL/v1/captures/$CAPTURE_ID" \
  -H "X-API-Key: $BLINDSIGHT_API_KEY"
```

Terminal success example:

```json
{
  "capture_id": "cap_01J67J4JYG9X5Q20MZB6S8N7P3",
  "scene_session_id": "ses_01J67J4K0BT3WD5H1RV2N7Q6XM",
  "source": {"type": "live", "mime_type": "video/webm"},
  "status": "succeeded",
  "card": {
    "capture_id": "cap_01J67J4JYG9X5Q20MZB6S8N7P3",
    "scene_session_id": "ses_01J67J4K0BT3WD5H1RV2N7Q6XM",
    "revision": 1,
    "evidence": ["cap_01J67J4JYG9X5Q20MZB6S8N7P3"],
    "card": {
      "place_type": "small shared bedroom",
      "place_type_confidence": "high",
      "overview": "The capture showed a shared bedroom with three beds, a study desk beside the wardrobes, and one person resting under a blanket. Pale walls and bright overhead lighting make the room visually plain and evenly lit.",
      "layout": [
        {
          "thing": "study desk",
          "relationship": "beside the wardrobes",
          "distance": "middle",
          "confidence": "high"
        },
        {
          "thing": "exit door",
          "relationship": "beyond the wardrobes",
          "distance": "far",
          "confidence": "medium"
        }
      ],
      "open_space": "A narrow clear strip runs between the beds and desk.",
      "people": [
        {
          "count_description": "one person",
          "relationship": "on the nearest bed",
          "activity": "resting under a blanket",
          "confidence": "high"
        }
      ],
      "visual_character": "Pale walls, dark furniture, and bright overhead lighting.",
      "uncertainties": [
        {
          "claim": "The far opening is the exit door.",
          "detail": "Only part of its frame was visible in the capture."
        }
      ]
    }
  },
  "failure": null,
  "created_at": "2026-08-29T11:30:00Z",
  "updated_at": "2026-08-29T11:30:09Z"
}
```

## Stage 0: preloaded excerpt

List excerpts:

```bash
curl --fail-with-body \
  "$BLINDSIGHT_URL/v1/excerpts" \
  -H "X-API-Key: $BLINDSIGHT_API_KEY"
```

Start one through the same capture resource:

```bash
curl --fail-with-body \
  -X POST "$BLINDSIGHT_URL/v1/captures" \
  -H "X-API-Key: $BLINDSIGHT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"source":{"type":"excerpt","excerpt_id":"via-001-entry-02"}}'
```

No chunks or completion call are needed. Poll the returned capture URL normally.

## Stage 1: card-first question and consent

Submit a question:

```bash
SESSION_ID="ses_01J67J4K0BT3WD5H1RV2N7Q6XM"

curl --fail-with-body \
  -X POST "$BLINDSIGHT_URL/v1/scene-sessions/$SESSION_ID/questions" \
  -H "X-API-Key: $BLINDSIGHT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question":"What colour was the mug on the desk?"}'
```

Poll the question URL. If the card lacks the detail, the terminal card-first result is:

```json
{
  "question_id": "que_01J67K73MPCZG9HTV4A2F6Y1WR",
  "scene_session_id": "ses_01J67J4K0BT3WD5H1RV2N7Q6XM",
  "question": "What colour was the mug on the desk?",
  "status": "needs_clip_consent",
  "answer": null,
  "source": null,
  "failure": null,
  "created_at": "2026-08-29T11:32:00Z",
  "updated_at": "2026-08-29T11:32:01Z"
}
```

After the client asks the user and receives explicit consent:

```bash
QUESTION_ID="que_01J67K73MPCZG9HTV4A2F6Y1WR"

curl --fail-with-body \
  -X POST "$BLINDSIGHT_URL/v1/scene-sessions/$SESSION_ID/questions/$QUESTION_ID/clip-check" \
  -H "X-API-Key: $BLINDSIGHT_API_KEY"
```

Poll the same question URL. A second miss settles as `status: unanswerable`, `answer: null`, and
`source: captured_view`. The client says, “I couldn’t tell from the capture.”

End the scene session:

```bash
curl --fail-with-body \
  -X DELETE "$BLINDSIGHT_URL/v1/scene-sessions/$SESSION_ID" \
  -H "X-API-Key: $BLINDSIGHT_API_KEY"
```

## Browser or React Native `fetch`

```js
const baseUrl = "https://example--blindsight-web.modal.run";
const apiKey = await loadApiKeySecurely();

async function api(path, options = {}) {
  const response = await fetch(`${baseUrl}${path}`, {
    ...options,
    headers: {
      "X-API-Key": apiKey,
      ...(options.headers || {}),
    },
  });

  if (!response.ok) {
    const body = await response.json();
    const error = new Error(body.error.message);
    error.code = body.error.code;
    error.retryable = body.error.retryable;
    error.details = body.error.details;
    throw error;
  }

  return response.status === 204 ? null : response.json();
}

async function waitForCapture(captureId) {
  const deadline = Date.now() + 90_000;
  while (Date.now() < deadline) {
    const capture = await api(`/v1/captures/${captureId}`);
    if (capture.status === "succeeded") return capture.card;
    if (capture.status === "failed") {
      const error = new Error(capture.failure.message);
      error.code = capture.failure.code;
      throw error;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw Object.assign(new Error("Capture did not settle within 90 seconds"), {
    code: "CLIENT_POLL_TIMEOUT",
  });
}

async function beginLiveCapture(mimeType = "video/webm") {
  return api("/v1/captures", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source: { type: "live", mime_type: mimeType } }),
  });
}

async function uploadChunk(captureId, index, blob) {
  return api(`/v1/captures/${captureId}/chunks/${index}`, {
    method: "PUT",
    headers: { "Content-Type": "application/octet-stream" },
    body: blob,
  });
}

async function completeCapture(captureId, chunkCount, mimeType = "video/webm") {
  await api(`/v1/captures/${captureId}/complete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chunk_count: chunkCount, mime_type: mimeType }),
  });
  return waitForCapture(captureId);
}
```

The browser reference client must call these same operations. It may not read backend memory or use
an undocumented same-process shortcut.

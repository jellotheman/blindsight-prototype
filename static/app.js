// The reference web client. It calls only the documented /v1 HTTP interface -- no in-process
// shortcut into the backend -- exactly like any other caller (see docs/spec/examples.md).

const KEY_STORAGE = "blindsight_api_key";
const CAPTURE_MS = 8000;
const CHUNK_TIMESLICE_MS = 1000;
const PULSE_START_MS = 3000;
const PULSE_INTERVAL_MS = 2000;
const STILL_WORKING_MS = 8000;
const DEFAULT_POLL_MS = 1000;
// Longer than the backend's default processing deadline (90s, see blindsight/app.py
// processing_deadline_seconds) so a slow-but-successful response never races a spurious
// client-side timeout right as the server is about to settle.
const POLL_DEADLINE_MS = 105000;
// Likewise longer than the backend's default question deadline (60s, see blindsight/app.py
// question_processing_deadline_seconds).
const QUESTION_POLL_DEADLINE_MS = 75000;

const CAPTURE_SETTLED_STATUSES = ["succeeded", "failed"];
const QUESTION_SETTLED_STATUSES = ["answered", "needs_clip_consent", "unanswerable", "failed"];

// The card had no grounds for an answer. The offer is spoken before any captured-view check is
// requested, and it warns about the extra wait the user is being asked to accept.
const CONSENT_OFFER =
  "I couldn't answer that from the scene card. Shall I check the captured view again? " +
  "This may take several seconds.";
// A second miss is a plain abstention. It is never turned into a confident negative claim.
const ABSTENTION = "I couldn't tell from the capture.";

const CANDIDATE_MIME_TYPES = [
  { recorder: "video/webm;codecs=vp9", contract: "video/webm" },
  { recorder: "video/webm;codecs=vp8", contract: "video/webm" },
  { recorder: "video/webm", contract: "video/webm" },
  { recorder: "video/mp4", contract: "video/mp4" },
];

// Live target bitrate for the recorder. The provider accepts ~750 kbps excerpts, and live WebM
// at the recorder default produced 3.4-8.7 MB per 8-second clip.
const LIVE_VIDEO_BITS_PER_SECOND = 1_000_000;

function getApiKey() {
  return window.localStorage.getItem(KEY_STORAGE) || "";
}

function setApiKey(value) {
  window.localStorage.setItem(KEY_STORAGE, value);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "X-API-Key": getApiKey(), ...(options.headers || {}) },
  });
  if (!response.ok) {
    const body = await response.json();
    throw Object.assign(new Error(body.error.message), {
      code: body.error.code,
      retryable: body.error.retryable,
      details: body.error.details,
    });
  }
  return response;
}

async function loadPosterObjectUrl(posterUrl) {
  const response = await api(posterUrl);
  return URL.createObjectURL(await response.blob());
}

function pickMime() {
  for (const candidate of CANDIDATE_MIME_TYPES) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(candidate.recorder)) {
      return candidate;
    }
  }
  throw new Error("This browser has no supported live-capture video format.");
}

// --- Speech -----------------------------------------------------------------------------------

function say(text) {
  return new Promise((resolve) => {
    if (!window.speechSynthesis) {
      resolve();
      return;
    }
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.onend = () => resolve();
    utterance.onerror = () => resolve();
    window.speechSynthesis.speak(utterance);
  });
}

// iOS will not speak unless synthesis was first triggered synchronously inside a user gesture,
// so every capture-starting tap spends part of itself on this before anything asynchronous runs.
function unlockSpeech() {
  if (!window.speechSynthesis) return;
  const utterance = new SpeechSynthesisUtterance(" ");
  utterance.volume = 0;
  window.speechSynthesis.speak(utterance);
}

// --- Earcons ------------------------------------------------------------------------------------
// A distinct, non-overlapping tone per meaning (see docs/spec/phase-0-1.md "Client requirements").
// The buzz is deliberately non-musical: a sawtooth wave reads as a fault, not a chime.

let audioCtx = null;

function getAudioContext() {
  if (!audioCtx) {
    const Ctor = window.AudioContext || window.webkitAudioContext;
    audioCtx = new Ctor();
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }
  return audioCtx;
}

function tone(frequency, durationMs, { type = "sine", gain = 0.2, delayMs = 0 } = {}) {
  const ctx = getAudioContext();
  const oscillator = ctx.createOscillator();
  const envelope = ctx.createGain();
  oscillator.type = type;
  oscillator.frequency.value = frequency;
  oscillator.connect(envelope);
  envelope.connect(ctx.destination);

  const startAt = ctx.currentTime + delayMs / 1000;
  const durationS = durationMs / 1000;
  envelope.gain.setValueAtTime(0, startAt);
  envelope.gain.linearRampToValueAtTime(gain, startAt + Math.min(0.01, durationS / 4));
  envelope.gain.linearRampToValueAtTime(0, startAt + durationS);
  oscillator.start(startAt);
  oscillator.stop(startAt + durationS + 0.02);
}

function playReadyEarcon() {
  tone(880, 150);
}

function playMetronomeTick() {
  tone(1200, 60, { gain: 0.15 });
}

function playCapturedEarcon() {
  tone(660, 110);
  tone(440, 140, { delayMs: 120 });
}

function playPulseEarcon() {
  tone(300, 80, { gain: 0.05 });
}

function playSettledEarcon() {
  tone(660, 100);
  tone(990, 160, { delayMs: 110 });
}

function playFailureBuzz() {
  tone(140, 500, { type: "sawtooth", gain: 0.25 });
}

// --- Wait ladder --------------------------------------------------------------------------------
// Processing begins in silence; a quiet pulse starts at three seconds; "Still working." is spoken
// once at eight seconds. See docs/spec/phase-0-1.md "Client requirements".

function startWaitLadder() {
  const state = { pulseInterval: null };
  const pulseTimeout = setTimeout(() => {
    playPulseEarcon();
    state.pulseInterval = setInterval(playPulseEarcon, PULSE_INTERVAL_MS);
  }, PULSE_START_MS);
  const stillWorkingTimer = setTimeout(() => say("Still working."), STILL_WORKING_MS);
  return {
    stop() {
      clearTimeout(pulseTimeout);
      if (state.pulseInterval) clearInterval(state.pulseInterval);
      clearTimeout(stillWorkingTimer);
    },
  };
}

// --- Polling -------------------------------------------------------------------------------------

function retryAfterMs(response) {
  const header = response.headers.get("Retry-After");
  const seconds = header ? Number(header) : NaN;
  return Number.isFinite(seconds) && seconds > 0 ? seconds * 1000 : DEFAULT_POLL_MS;
}

// Captures and questions are the same asynchronous resource shape -- poll the Location URL,
// respect Retry-After -- and differ only in which statuses count as settled.
async function pollUntilSettled(location, settledStatuses, deadlineMs) {
  const deadline = Date.now() + deadlineMs;
  for (;;) {
    const response = await api(location);
    const resource = await response.json();
    if (settledStatuses.includes(resource.status)) {
      return resource;
    }
    if (Date.now() > deadline) {
      throw Object.assign(new Error("The request did not settle in time."), {
        code: "CLIENT_POLL_TIMEOUT",
      });
    }
    await new Promise((resolve) => setTimeout(resolve, retryAfterMs(response)));
  }
}

// Shared by live and excerpt sources once the capture resource exists and is being processed:
// the wait ladder, settlement, and spoken success/failure flow are identical from here on.
async function runToSettlement(location, statusEl, sceneSessionId) {
  const ladder = startWaitLadder();
  try {
    const settled = await pollUntilSettled(location, CAPTURE_SETTLED_STATUSES, POLL_DEADLINE_MS);
    ladder.stop();
    if (settled.status === "succeeded") {
      const overview = settled.card.card.overview;
      statusEl.textContent = overview;
      // The scene card is the grounding for every follow-up question until this session ends.
      beginSceneSession(settled.scene_session_id);
      await say(overview);
      playSettledEarcon();
    } else {
      statusEl.textContent = settled.failure.message;
      playFailureBuzz();
      // A capture that never produced a card leaves a scene session no one will ever ask about.
      // The backend cannot infer that, so the client that opened it has to close it.
      await deleteSceneSession(sceneSessionId);
    }
    return settled;
  } catch (err) {
    ladder.stop();
    statusEl.textContent = err.message;
    playFailureBuzz();
    await deleteSceneSession(sceneSessionId);
    return null;
  }
}

// --- Live capture ---------------------------------------------------------------------------------

let liveStream = null;

async function ensureLiveStream() {
  if (liveStream && liveStream.active) return liveStream;
  liveStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment" },
    audio: false,
  });
  return liveStream;
}

async function runLiveCapture(statusEl) {
  const mime = pickMime();

  statusEl.textContent = "Requesting camera access...";
  const stream = await ensureLiveStream();

  const created = await (
    await api("/v1/captures", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: { type: "live", mime_type: mime.contract } }),
    })
  ).json();
  const location = `/v1/captures/${created.capture_id}`;

  playReadyEarcon();
  statusEl.textContent = "Look around at what you'd like described.";
  await say("Look around at what you'd like described.");

  const recorder = new MediaRecorder(stream, {
    mimeType: mime.recorder,
    videoBitsPerSecond: LIVE_VIDEO_BITS_PER_SECOND,
  });
  const uploads = [];
  let nextIndex = 0;
  // MediaRecorder's streaming WebM carries no Duration element in the Segment Info, and the
  // provider rejects such files. The Info lives in chunk 0, so chunk 0 is held back here and
  // patched with the measured wall-clock duration once recording ends, then uploaded like any
  // other chunk (the patch is a header rewrite; chunks 1+ upload concurrently as before). If
  // patching fails, the unpatched chunk 0 is uploaded as-is -- backend validation still guards.
  let chunk0 = null;
  recorder.ondataavailable = (event) => {
    if (event.data.size === 0) return;
    const index = nextIndex++;
    if (index === 0) {
      chunk0 = event.data;
      return;
    }
    uploads.push(
      api(`/v1/captures/${created.capture_id}/chunks/${index}`, {
        method: "PUT",
        headers: { "Content-Type": "application/octet-stream" },
        body: event.data,
      })
    );
  };

  const stopped = new Promise((resolve) => {
    recorder.onstop = resolve;
  });
  const captureStartedAt = performance.now();
  recorder.start(CHUNK_TIMESLICE_MS);
  statusEl.textContent = "Recording...";
  const metronome = setInterval(playMetronomeTick, 1000);
  playMetronomeTick();

  await new Promise((resolve) => setTimeout(resolve, CAPTURE_MS));
  clearInterval(metronome);
  recorder.stop();
  playCapturedEarcon();
  await stopped;
  const captureDurationMs = performance.now() - captureStartedAt;

  if (chunk0) {
    let uploadBlob = chunk0;
    if (mime.contract === "video/webm" && window.fixWebmDurationLib) {
      try {
        uploadBlob = await window.fixWebmDurationLib.fixWebmDuration(chunk0, captureDurationMs);
      } catch {
        // Keep the unpatched chunk 0; the provider rejection is handled downstream.
      }
    }
    uploads.push(
      api(`/v1/captures/${created.capture_id}/chunks/0`, {
        method: "PUT",
        headers: { "Content-Type": "application/octet-stream" },
        body: uploadBlob,
      })
    );
  }
  await Promise.all(uploads);
  statusEl.textContent = "Processing...";
  await api(`/v1/captures/${created.capture_id}/complete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chunk_count: nextIndex, mime_type: mime.contract }),
  });

  return runToSettlement(location, statusEl, created.scene_session_id);
}

// --- Excerpt selection ------------------------------------------------------------------------

async function runExcerptCapture(excerptId, statusEl) {
  statusEl.textContent = "Processing...";
  const created = await (
    await api("/v1/captures", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: { type: "excerpt", excerpt_id: excerptId } }),
    })
  ).json();
  return runToSettlement(
    `/v1/captures/${created.capture_id}`,
    statusEl,
    created.scene_session_id
  );
}

async function renderExcerpts() {
  const list = document.getElementById("excerpts");
  list.textContent = "";

  let items;
  try {
    items = (await (await api("/v1/excerpts")).json()).items;
  } catch (err) {
    const li = document.createElement("li");
    li.textContent = `Could not load excerpts: ${err.message}`;
    list.appendChild(li);
    return;
  }

  for (const excerpt of items) {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    const img = document.createElement("img");
    img.alt = excerpt.label;
    loadPosterObjectUrl(excerpt.poster_url).then((url) => (img.src = url));
    const label = document.createElement("span");
    label.textContent = `${excerpt.label} (${excerpt.duration_seconds}s)`;
    button.append(img, label);
    button.addEventListener("click", () => handleExcerptTap(excerpt.excerpt_id));
    li.appendChild(button);
    list.appendChild(li);
  }
}

// --- Stage 1 scene session and follow-up questions ----------------------------------------------
// A scene session is one scene card and the conversation grounded in it. "Done," a new capture, or
// leaving the page ends it, and the client says so explicitly: the shared key carries no client
// identity, so the backend cannot infer which earlier scene session a new capture supersedes.

let activeSceneSessionId = null;
// Set only while the backend has settled a question as needs_clip_consent and the user has been
// asked. Nothing may reach the captured-view check without passing through this.
let pendingConsent = null;

function beginSceneSession(sceneSessionId) {
  activeSceneSessionId = sceneSessionId;
  hideConsentPrompt();
  document.getElementById("answer").textContent = "";
  document.getElementById("question").value = "";
  document.getElementById("conversation").hidden = false;
}

async function deleteSceneSession(sceneSessionId) {
  if (!sceneSessionId) return;
  try {
    await api(`/v1/scene-sessions/${sceneSessionId}`, { method: "DELETE" });
  } catch {
    // The session is over for this client either way, and a new capture must not be blocked by a
    // failed teardown of the one it supersedes.
  }
}

function closeConversationSurface() {
  activeSceneSessionId = null;
  hideConsentPrompt();
  document.getElementById("conversation").hidden = true;
  document.getElementById("answer").textContent = "";
}

async function endActiveSceneSession() {
  const sceneSessionId = activeSceneSessionId;
  closeConversationSurface();
  await deleteSceneSession(sceneSessionId);
}

function showConsentPrompt(questionId, sceneSessionId) {
  pendingConsent = { questionId, sceneSessionId };
  document.getElementById("consent-offer").textContent = CONSENT_OFFER;
  document.getElementById("consent").hidden = false;
}

function hideConsentPrompt() {
  pendingConsent = null;
  document.getElementById("consent").hidden = true;
  document.getElementById("consent-offer").textContent = "";
}

// One place where a settled question becomes speech, shared by the card answer and the
// captured-view answer. A miss is spoken as a miss; it never becomes a confident negative.
async function speakQuestionOutcome(settled) {
  const answerEl = document.getElementById("answer");
  if (settled.status === "answered") {
    answerEl.textContent = settled.answer;
    await say(settled.answer);
    playSettledEarcon();
  } else if (settled.status === "needs_clip_consent") {
    answerEl.textContent = "";
    showConsentPrompt(settled.question_id, settled.scene_session_id);
    await say(CONSENT_OFFER);
  } else if (settled.status === "unanswerable") {
    answerEl.textContent = ABSTENTION;
    await say(ABSTENTION);
  } else {
    answerEl.textContent = settled.failure.message;
    playFailureBuzz();
  }
}

async function pollQuestionToSettlement(location) {
  const ladder = startWaitLadder();
  try {
    return await pollUntilSettled(location, QUESTION_SETTLED_STATUSES, QUESTION_POLL_DEADLINE_MS);
  } finally {
    ladder.stop();
  }
}

async function askQuestion(question) {
  const sceneSessionId = activeSceneSessionId;
  const created = await (
    await api(`/v1/scene-sessions/${sceneSessionId}/questions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    })
  ).json();
  const location = `/v1/scene-sessions/${sceneSessionId}/questions/${created.question_id}`;
  await speakQuestionOutcome(await pollQuestionToSettlement(location));
}

async function handleAsk() {
  if (busy || !activeSceneSessionId) return;
  const input = document.getElementById("question");
  const question = input.value.trim();
  if (!question) return;
  unlockSpeech();
  getAudioContext();
  hideConsentPrompt();
  setBusy(true);
  const answerEl = document.getElementById("answer");
  try {
    input.value = "";
    await askQuestion(question);
  } catch (err) {
    answerEl.textContent = err.message;
    playFailureBuzz();
  } finally {
    setBusy(false);
  }
}

// The only path to the captured-view check. Reaching it means the user heard the offer, including
// its wait warning, and agreed.
async function handleConsentAgreement() {
  if (busy || !pendingConsent) return;
  const { questionId, sceneSessionId } = pendingConsent;
  unlockSpeech();
  hideConsentPrompt();
  setBusy(true);
  const answerEl = document.getElementById("answer");
  const location = `/v1/scene-sessions/${sceneSessionId}/questions/${questionId}`;
  try {
    await api(`${location}/clip-check`, { method: "POST" });
    await speakQuestionOutcome(await pollQuestionToSettlement(location));
  } catch (err) {
    answerEl.textContent = err.message;
    playFailureBuzz();
  } finally {
    setBusy(false);
  }
}

async function handleConsentRefusal() {
  if (busy || !pendingConsent) return;
  unlockSpeech();
  hideConsentPrompt();
  // Declining leaves the question unanswered. The acknowledgement confirms the control was heard
  // without saying anything about the scene: that would be a claim the evidence was never
  // checked for.
  document.getElementById("answer").textContent = "";
  await say("All right.");
}

async function handleDone() {
  if (busy || !activeSceneSessionId) return;
  unlockSpeech();
  setBusy(true);
  try {
    await endActiveSceneSession();
    // A control that ends the conversation has to be audible; the user cannot see it happen.
    await say("Done.");
  } finally {
    setBusy(false);
  }
}

// --- Orchestration --------------------------------------------------------------------------------

let busy = false;

function setBusy(value) {
  busy = value;
  document.getElementById("tap-target").setAttribute("aria-disabled", String(value));
  for (const button of document.querySelectorAll("#excerpts button")) {
    button.disabled = value;
  }
  for (const id of ["ask", "done", "question", "consent-yes", "consent-no"]) {
    document.getElementById(id).disabled = value;
  }
}

async function handleTap() {
  if (busy) return;
  unlockSpeech();
  getAudioContext();
  setBusy(true);
  const statusEl = document.getElementById("status");
  try {
    await endActiveSceneSession();
    await runLiveCapture(statusEl);
  } catch (err) {
    statusEl.textContent = err.message;
    playFailureBuzz();
  } finally {
    setBusy(false);
  }
}

async function handleExcerptTap(excerptId) {
  if (busy) return;
  unlockSpeech();
  getAudioContext();
  setBusy(true);
  const statusEl = document.getElementById("status");
  try {
    await endActiveSceneSession();
    await runExcerptCapture(excerptId, statusEl);
  } catch (err) {
    statusEl.textContent = err.message;
    playFailureBuzz();
  } finally {
    setBusy(false);
  }
}

function showApp() {
  document.getElementById("key-gate").hidden = true;
  document.getElementById("app").hidden = false;
  renderExcerpts();
}

function showKeyGate() {
  document.getElementById("api-key").value = getApiKey();
  document.getElementById("key-gate").hidden = false;
  document.getElementById("app").hidden = true;
}

document.getElementById("save-key").addEventListener("click", () => {
  const value = document.getElementById("api-key").value.trim();
  if (!value) return;
  setApiKey(value);
  showApp();
});

document.getElementById("change-key").addEventListener("click", showKeyGate);
document.getElementById("tap-target").addEventListener("click", handleTap);

document.getElementById("ask").addEventListener("click", handleAsk);
document.getElementById("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter") handleAsk();
});
document.getElementById("consent-yes").addEventListener("click", handleConsentAgreement);
document.getElementById("consent-no").addEventListener("click", handleConsentRefusal);
document.getElementById("done").addEventListener("click", handleDone);

// Application shutdown ends the scene session too. keepalive lets the DELETE outlive the page.
// event.persisted marks a page going into the back/forward cache rather than away: on a phone
// that is an app switch, a screen lock or a call, and the user expects to come back to the same
// conversation. Ending the session there would leave a live-looking panel whose controls no
// longer do anything -- indistinguishable, with the screen off, from a broken app.
window.addEventListener("pagehide", (event) => {
  if (event.persisted || !activeSceneSessionId) return;
  fetch(`/v1/scene-sessions/${activeSceneSessionId}`, {
    method: "DELETE",
    headers: { "X-API-Key": getApiKey() },
    keepalive: true,
  });
  activeSceneSessionId = null;
});

// A restored page keeps the DOM it was frozen with, so a surface left open without a session
// behind it has to be closed rather than silently swallowing taps.
window.addEventListener("pageshow", (event) => {
  if (event.persisted && !activeSceneSessionId) closeConversationSurface();
});

if (getApiKey()) {
  showApp();
} else {
  showKeyGate();
}

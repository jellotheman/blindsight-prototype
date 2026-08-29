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

const CANDIDATE_MIME_TYPES = [
  { recorder: "video/webm;codecs=vp9", contract: "video/webm" },
  { recorder: "video/webm;codecs=vp8", contract: "video/webm" },
  { recorder: "video/webm", contract: "video/webm" },
  { recorder: "video/mp4", contract: "video/mp4" },
];

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

async function pollUntilSettled(location) {
  const deadline = Date.now() + POLL_DEADLINE_MS;
  for (;;) {
    const response = await api(location);
    const resource = await response.json();
    if (resource.status === "succeeded" || resource.status === "failed") {
      return resource;
    }
    if (Date.now() > deadline) {
      throw Object.assign(new Error("The capture did not settle in time."), {
        code: "CLIENT_POLL_TIMEOUT",
      });
    }
    await new Promise((resolve) => setTimeout(resolve, retryAfterMs(response)));
  }
}

// Shared by live and excerpt sources once the capture resource exists and is being processed:
// the wait ladder, settlement, and spoken success/failure flow are identical from here on.
async function runToSettlement(location, statusEl) {
  const ladder = startWaitLadder();
  try {
    const settled = await pollUntilSettled(location);
    ladder.stop();
    if (settled.status === "succeeded") {
      const overview = settled.card.card.overview;
      statusEl.textContent = overview;
      await say(overview);
      playSettledEarcon();
    } else {
      statusEl.textContent = settled.failure.message;
      playFailureBuzz();
    }
    return settled;
  } catch (err) {
    ladder.stop();
    statusEl.textContent = err.message;
    playFailureBuzz();
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

  const recorder = new MediaRecorder(stream, { mimeType: mime.recorder });
  const uploads = [];
  let nextIndex = 0;
  recorder.ondataavailable = (event) => {
    if (event.data.size === 0) return;
    const index = nextIndex++;
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
  recorder.start(CHUNK_TIMESLICE_MS);
  statusEl.textContent = "Recording...";
  const metronome = setInterval(playMetronomeTick, 1000);
  playMetronomeTick();

  await new Promise((resolve) => setTimeout(resolve, CAPTURE_MS));
  clearInterval(metronome);
  recorder.stop();
  playCapturedEarcon();
  await stopped;

  await Promise.all(uploads);
  statusEl.textContent = "Processing...";
  await api(`/v1/captures/${created.capture_id}/complete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chunk_count: nextIndex, mime_type: mime.contract }),
  });

  return runToSettlement(location, statusEl);
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
  return runToSettlement(`/v1/captures/${created.capture_id}`, statusEl);
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

// --- Orchestration --------------------------------------------------------------------------------

let busy = false;

function setBusy(value) {
  busy = value;
  document.getElementById("tap-target").setAttribute("aria-disabled", String(value));
  for (const button of document.querySelectorAll("#excerpts button")) {
    button.disabled = value;
  }
}

async function handleTap() {
  if (busy) return;
  unlockSpeech();
  getAudioContext();
  setBusy(true);
  const statusEl = document.getElementById("status");
  try {
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

if (getApiKey()) {
  showApp();
} else {
  showKeyGate();
}

// The reference web client. It calls only the documented /v1 HTTP interface -- no in-process
// shortcut into the backend -- exactly like any other caller (see docs/spec/examples.md).

const KEY_STORAGE = "blindsight_api_key";

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
    throw Object.assign(new Error(body.error.message), { code: body.error.code });
  }
  return response;
}

async function loadPosterObjectUrl(posterUrl) {
  const response = await api(posterUrl);
  return URL.createObjectURL(await response.blob());
}

async function renderExcerpts() {
  const list = document.getElementById("excerpts");
  list.textContent = "";
  if (!getApiKey()) return;

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
    const img = document.createElement("img");
    img.alt = excerpt.label;
    img.width = 160;
    loadPosterObjectUrl(excerpt.poster_url).then((url) => (img.src = url));
    const label = document.createElement("span");
    label.textContent = `${excerpt.label} (${excerpt.duration_seconds}s)`;
    li.append(img, label);
    list.appendChild(li);
  }
}

document.getElementById("api-key").value = getApiKey();
document.getElementById("save-key").addEventListener("click", () => {
  setApiKey(document.getElementById("api-key").value.trim());
  renderExcerpts();
});

renderExcerpts();

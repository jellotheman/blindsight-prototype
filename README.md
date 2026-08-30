# BlindSight Prototype

[NOTE: INITIAL IMPLEMENTATION OF FIGMA PROTOTYPE IN REACT NATIVE, BY TEAM MEMBER AVIYANSH LOMEO]
- [https://github.com/aviyanshlomeo568-cpu/BlindSight]

[ACCESS WEB PREVIEW DEMO HOSTED ON MODAL]
- [https://jellotheman--blindsight-bench-web.modal.run]
- [https://jellotheman--blindsight-api-web.modal.run]

[NOTE: MODAL HOSTED LINKS SOMETIMES AREN'T CERTIFIED IN APPLE BASED BROWSERS, OR EVEN CHROME. UPON TESTING, WORKS IN FIREFOX. MY BAD FOR INCONVENIENCE, WILL FIX LATER.]

BlindSight is an environmental-understanding prototype for blind and low-vision people. A client
records a short, user-directed captured view, the backend returns a structured scene card, and the
client speaks a concise orientation before offering details on demand.

This repository is specification-first. The primary artifact is a text-only HTTP interface that the
Expo client in `frontend/`, the legacy browser reference client, and other native clients can use
equally.

## Start here

- [`docs/spec/phase-0-1.md`](docs/spec/phase-0-1.md) — product and implementation specification
- [`docs/spec/openapi.yaml`](docs/spec/openapi.yaml) — complete machine-readable HTTP contract
- [`docs/spec/examples.md`](docs/spec/examples.md) — worked `curl` and `fetch` flows
- [`CONTEXT.md`](CONTEXT.md) — canonical product vocabulary
- [`docs/rollback.md`](docs/rollback.md) — how to get back to a known-working state
- `REFERENCE.local.md` — read this too if it is present in your checkout

## Scope

- **Stage 0:** an eight-second captured view produces a validated scene card.
- **Stage 1:** follow-up questions use the scene card first and may re-check the stored capture only
  after explicit consent.

Stage 1 is the explicit build cut line. Stage 0 must remain independently useful.

BlindSight is not a navigation or mobility aid and makes no safety claim.

## Status

The contract is ready to drive implementation. The first implementation target is the Stage 0
contract test suite described in the specification.

## Run locally, reachable from a phone

The two normal startup paths are:

```powershell
.\start-web.ps1          # exported web app, available only on this computer
.\start-web.ps1 -Phone   # exported web app in a phone browser
.\start-expo.ps1         # native app through Expo Go or a development build
```

Both scripts prepare a lightweight local Python environment and install dependencies when needed.
They generate temporary API keys at runtime; no key or environment configuration is stored in
either script. `start-expo.ps1` opens the API in a second terminal and starts Expo in the first.
Enter the API terminal's HTTPS URL and temporary key in BlindSight's Settings screen. Phone modes
require `cloudflared` on `PATH`; Expo's tunnel may also install or request its tunnel helper.

The manual setup is available when you need finer control:

Install and export the Expo web client before starting the Python server:

```powershell
cd frontend
npm ci
npm run build:web
cd ..
```

The Expo app is then served at `/`; the legacy reference web client remains available at
`/reference/`. Camera, speech, and the audio ladder are only honestly testable from a real phone.
One command starts the backend locally and publishes it through a Cloudflare quick tunnel, exposing
the identical `/v1` interface a phone would reach on Modal:

```powershell
python -m tools.local_dev --api-key <any shared key you choose>
```

This requires `cloudflared` on `PATH` (<https://developers.cloudflare.com/cloudflared/downloads/>)
and the `dev` extra installed (`pip install -e ".[dev]"`, which brings in `uvicorn`). It fails
loudly if `cloudflared` is missing, the port is already in use, or no tunnel URL appears within 30
seconds, and stops both the server and the tunnel on Ctrl+C. Requests need no change beyond the
base URL between this tunnel and the Modal deployment. It uses the same in-memory store and
deterministic provider as the walking-skeleton defaults; it does not exercise live Reka/Gemini,
which belongs to issue #10.

## Run on Modal

Modal authentication and application authentication are separate:

- The Modal CLI uses the local Modal profile created by `modal token new`. GitHub Actions uses the
  repository secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` instead.
- The BlindSight HTTP API uses the `BLINDSIGHT_API_KEY` value stored in the Modal secret named
  `blindsight-api-key`. Enter that value in the client's Settings screen; it is sent as `X-API-Key`
  and stored through the platform credential adapter where available.

For a live-reloading development URL, run:

```powershell
python -m modal serve modal_app.py
```

The function declaration in `modal_app.py` attaches `blindsight-api-key`, so `serve` and `deploy`
both receive `BLINDSIGHT_API_KEY` without placing it in source control or a local environment file.

Production capture processing also requires a Modal secret named `blindsight-provider-keys` with
`REKA_API_KEY` and `GEMINI_API_KEY`. Reka defaults to the stable `reka-flash` alias; override it
with `BLINDSIGHT_REKA_MODEL` in that secret only after verifying the replacement model. Retained
captures, raw attempts, usage, timings, selections, cards, and failures are written to the
`blindsight-evidence` Modal Volume.

Download retained run directories when re-judging a prompt or schema, then replay one or a set
without changing the accepted record:

```powershell
python -m tools.replay --evidence-root runs --provider gemini --capture-id cap_example
```

Reka replay additionally needs the deployed HTTPS base URL so its short-lived media token can be
stored in the shared Modal Dict and fetched by Reka.

To update the persistent deployment manually, run:

```powershell
cd frontend
npm ci
npm run build:web
cd ..
python -m modal deploy modal_app.py
```

Every push to `main` also runs the test suite and, if it passes, deploys the application through
`.github/workflows/deploy-modal.yml`. The workflow requires the two Modal token repository secrets
described above; the application API key remains in Modal and is not copied into GitHub.

## License

MIT. See [`LICENSE`](LICENSE).

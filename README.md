# BlindSight Prototype

BlindSight is an environmental-understanding prototype for blind and low-vision people. A client
records a short, user-directed captured view, the backend returns a structured scene card, and the
client speaks a concise orientation before offering details on demand.

This repository is specification-first. The primary artifact is a text-only HTTP interface that a
browser reference client and a React Native Android client can use equally.

## Start here

- [`docs/spec/phase-0-1.md`](docs/spec/phase-0-1.md) — product and implementation specification
- [`docs/spec/openapi.yaml`](docs/spec/openapi.yaml) — complete machine-readable HTTP contract
- [`docs/spec/examples.md`](docs/spec/examples.md) — worked `curl` and `fetch` flows
- [`CONTEXT.md`](CONTEXT.md) — canonical product vocabulary
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

## Run on Modal

Modal authentication and application authentication are separate:

- The Modal CLI uses the local Modal profile created by `modal token new`. GitHub Actions uses the
  repository secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` instead.
- The BlindSight HTTP API uses the `BLINDSIGHT_API_KEY` value stored in the Modal secret named
  `blindsight-api-key`. Enter that value in the reference client; it is sent as `X-API-Key` and
  retained only in that browser's local storage.

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
python -m modal deploy modal_app.py
```

Every push to `main` also runs the test suite and, if it passes, deploys the application through
`.github/workflows/deploy-modal.yml`. The workflow requires the two Modal token repository secrets
described above; the application API key remains in Modal and is not copied into GitHub.

## License

MIT. See [`LICENSE`](LICENSE).

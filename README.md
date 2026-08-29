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

## Scope

- **Stage 0:** an eight-second captured view produces a validated scene card.
- **Stage 1:** follow-up questions use the scene card first and may re-check the stored capture only
  after explicit consent.

Stage 1 is the explicit build cut line. Stage 0 must remain independently useful.

BlindSight is not a navigation or mobility aid and makes no safety claim.

## Status

The contract is ready to drive implementation. The first implementation target is the Stage 0
contract test suite described in the specification.

## License

MIT. See [`LICENSE`](LICENSE).

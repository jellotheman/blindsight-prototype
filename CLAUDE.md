## Source of truth

Read `CONTEXT.md`, `docs/spec/phase-0-1.md`, and `docs/spec/openapi.yaml` before planning or
implementing. The OpenAPI document is the public interface. If prose and OpenAPI disagree, stop and
correct the specification before coding.

Also read `REFERENCE.local.md` if it is present in the repository root. It is untracked and may be
absent; its absence is not an error, and it never overrides the specification.

## Test seam

Test through the public HTTP interface using real request/response serialization and deterministic
provider and job-store adapters. Assert observable behavior, not module internals. Live provider
calls are smoke tests, not the acceptance suite.

## Product constraints

- Compose released components; train nothing.
- Make no safety, navigation, hazard-detection, or mobility-aid claim.
- Treat uncertainty as structured claim-specific data, not a disclaimer.
- Keep Stage 0 useful without Stage 1.
- The HTTP interface is text-only. Clients own speech and interaction sounds.
- Do not introduce persistent place memory or imply complete/current awareness.

## Issue tracker

Issues live in this repository's GitHub Issues. Fully specified implementation issues use the
`ready-for-agent` label.

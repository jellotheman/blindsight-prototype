# Getting back to a known-working state

This is the recovery procedure for local work or a deployment going sideways. It assumes the
`main` branch is always the last known-good state (every push to `main` runs the test suite and
auto-deploys — see the README's "Run on Modal" section).

## Checkpoints

| Commit | What it captures |
| --- | --- |
| `4268db9` | Working state as of 2026-08-31: Modal `web()` no longer wires `ModalMediaUrlStore`/`BLINDSIGHT_PUBLIC_BASE_URL`, since Gemini takes captured video inline as base64 and Reka is out of the loop. |

Add a row here whenever you deliberately checkpoint a working state, with the commit hash and a
one-line reason. Find the latest good commit yourself with `git log --oneline` if this table is
stale — it is a pointer, not the source of truth.

## Local working tree

If uncommitted changes leave the app broken, first check what you'd lose:

```powershell
git status
git diff
```

Then discard back to the last commit (or stash instead of discarding if you might want the
changes later):

```powershell
git stash -u          # keep the changes, unstaged and out of the way
# or, to discard entirely:
git checkout -- .
git clean -fd
```

## Rolling back committed work

To move `main` back to a specific known-good commit (e.g. the checkpoint above):

```powershell
git checkout main
git reset --hard 4268db9   # replace with the commit you want to restore
```

Only do this on a branch nobody else has pulled from, or after checking with the team — it
rewrites history that a `git push --force` would then need to propagate. The safer alternative
that preserves history is a revert of the offending commit(s):

```powershell
git revert <bad-commit>
```

## Rolling back the Modal deployment

`main` auto-deploys via `.github/workflows/deploy-modal.yml` on every passing push, so the
fastest recovery path is usually: fix or revert on `main`, push, let CI redeploy.

To deploy a specific known-good commit manually instead of waiting on CI:

```powershell
git checkout 4268db9
python -m modal deploy modal_app.py
git checkout main
```

This overwrites the live Modal app with that commit's code. It does not touch the
`blindsight-evidence` Volume or the `blindsight-api-key` / `blindsight-provider-keys` secrets —
those are independent of which commit is deployed.

## Verifying the restored state

After rolling back, confirm the app actually works before moving on:

```powershell
python -m pytest
python -m tools.local_dev --api-key <any shared key you choose>
```

`tools.local_dev` exercises the same `/v1` interface Modal serves, using the in-memory store and
deterministic provider, so it will surface a broken `create_app`/`modal_app.py` wiring without
needing a live Modal deploy.

# Ego4D CLI: selective, reviewable retrieval

Research date: 2026-08-30. Sources below are official Ego4D package/docs or
the official EgoEnv repository.

## Pin and credentials

- Pin `ego4d==1.7.3`. The upstream package currently declares that version and
  exposes the `ego4d` console command. Use Python 3.10 or newer: the upstream
  root README is newer than the CLI README's legacy Python-3.8 minimum.
  ([package source](https://raw.githubusercontent.com/facebookresearch/Ego4d/main/setup.py),
  [upstream setup](https://github.com/facebookresearch/Ego4d#setup))
- Dataset access requires an accepted Ego4D licence and AWS credentials. The
  CLI constructs a boto3 session from `--aws_profile_name`, which defaults to
  `default`; configure a named profile and pass it explicitly when using the
  Modal secret. A missing profile is an error.
  ([Ego4D start guide](https://ego4d-data.org/docs/start-here/),
  [CLI README](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md),
  [configuration source](https://raw.githubusercontent.com/facebookresearch/Ego4d/main/ego4d/cli/config.py))

## Dataset and identifier choice

- EgoEnv's RoomPred tables have both `video_uid` and `clip_uid`; its own data
  guide says those clips should be downloaded with the official Ego4D CLI.
  ([EgoEnv dataset guide](https://raw.githubusercontent.com/facebookresearch/ego-env/main/DATASETS.md))
- The current CLI's video datasets are `full_scale`, `clips`, and
  `video_540ss`. `video_540ss` resizes the short side to 540 pixels and is the
  practical first choice for this corpus. `--video_uids` accepts video or clip
  UIDs; `--video_uid_file` accepts whitespace-delimited UIDs and is mutually
  exclusive with it. The filter applies only to video datasets, not
  `annotations`.
  ([CLI README](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md),
  [configuration](https://raw.githubusercontent.com/facebookresearch/Ego4d/main/ego4d/cli/config.py),
  [selection code](https://raw.githubusercontent.com/facebookresearch/Ego4d/main/ego4d/cli/download.py))
- Do not substitute `annotations_540ss`: it is described for spatial labels,
  and the current source does not include it in its known dataset list (though
  it will attempt unknown dataset names). The project needs the separately
  supplied EgoEnv labels, not Ego4D spatial annotations.
  ([CLI README](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md),
  [configuration](https://raw.githubusercontent.com/facebookresearch/Ego4d/main/ego4d/cli/config.py))

## Safe preview / size estimate

There is no true dry-run option. A normal run downloads metadata/manifests and
performs S3 object checks before it reports the summed expected GiB and asks
for confirmation. Declining (`n`) exits before `download_all`, so no video
media is transferred. Do **not** pass `-y`/`--yes`; it bypasses that gate. Do
not pass `--skip-s3-checks`, because it disables the size estimate.
([CLI flow](https://raw.githubusercontent.com/facebookresearch/Ego4d/main/ego4d/cli/cli.py),
[configuration](https://raw.githubusercontent.com/facebookresearch/Ego4d/main/ego4d/cli/config.py))

Recommended command shape (do not use it unattended):

```text
ego4d --output_directory <staging-root> --version v2_1 \
  --datasets video_540ss --video_uid_file <annotated-video-uids.txt> \
  --no-metadata --aws_profile_name <profile>
```

At the confirmation prompt, answer `n` to retain the downloaded manifest and
the terminal size estimate without downloading media. `--list-datasets` is an
even smaller remote check: it retrieves the catalogue then returns before an
output directory or dataset downloads, but still validates the AWS profile.
([CLI flow](https://raw.githubusercontent.com/facebookresearch/Ego4d/main/ego4d/cli/cli.py),
[manifest implementation](https://raw.githubusercontent.com/facebookresearch/Ego4d/main/ego4d/cli/manifest.py))

The CLI saves each dataset manifest and later records downloaded files in
`manifest.ver`, allowing it to skip unchanged local files on a subsequent
selective run. The corpus builder should therefore persist its UID file and
the CLI's returned manifest/estimate as the reproducible acquisition plan.
([download implementation](https://raw.githubusercontent.com/facebookresearch/Ego4d/main/ego4d/cli/download.py))

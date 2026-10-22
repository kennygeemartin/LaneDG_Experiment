# Published run metadata

`protocol.json`, `environment.json`, and `verification.json` describe the
completed real-image run. The protocol is a publication-safe copy of the
executed protocol: the machine-specific output directory was removed. Its
original protocol SHA-256 is retained as `original_execution_protocol_sha256`;
the sanitized file therefore has a different file hash. The local Python
executable path was likewise removed from the environment record.

The selected image manifests are in `../manifests/`. Dataset images,
annotations, checkpoints, masks, and per-image prediction maps are not
redistributed here. Re-download the licensed benchmark data using
`../DATA_SOURCES.md` and verify against the published manifest hashes before
running the pipeline.

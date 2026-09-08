# Wire vocabulary reference

This directory vendors six source files from [asqav-registry](https://github.com/jagmarques/asqav-registry/tree/79f5d28fd132c97bc4a5eae02a2bafce72b145af) at commit `79f5d28fd132c97bc4a5eae02a2bafce72b145af`. `upstream.json` records their SHA256 digests.

From the repository root, install the developer dependency and check the reference outputs:

```sh
python -m pip install -r tools/wire-vocabulary-requirements.txt
python tools/sync_wire_vocabulary.py --check
```

Use `--write` to regenerate the three files under `vendor/wire-vocabulary/vocabulary/generated/`. Both commands verify the pinned source inventory and bytes before loading the renderer, and work offline after dependency installation. `--verify-upstream` separately compares the six immutable public source files over HTTPS using curl; it changes no files.

These outputs are reference examples. Runtime consumers and distribution attribution require separate integration. Local hashes establish consistency with the reviewed manifest; changing a manifest is not authentication of a new upstream snapshot. Writes are sequential; rerun `--check` after a failed write.

The reference files retain their upstream [Apache License 2.0](LICENSE) and [NOTICE](NOTICE). The surrounding Asqav implementation remains under its repository license.

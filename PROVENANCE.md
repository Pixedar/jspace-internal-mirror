# Dated source and evidence

Creator: **Wiktor Tomasik**. Experiments began 4–5 November 2026. The companion comparison identifies the 3 November 2026 [signed external-reader edition](https://github.com/Pixedar/jspace-strange-loop/tree/study-2026-11-03) separately.

The signed edition tag and `PUBLICATION_RECORD.json` identify this release. The record contains SHA-256 checksums of the release files, the numerical deposit archive, and the original experiment records through the evidence manifest. GitHub's public release date records when this edition was shared. These records establish a dated public artifact and a verifiable link to its author. Priority relative to independent work requires separate evidence.

To verify the manifest signature from the repository root, using OpenSSH and a shell with byte-preserving input:

```sh
ssh-keygen -Y verify -f PUBLICATION_ALLOWED_SIGNERS -I pixedar@gmail.com \
  -n file -s PUBLICATION_RECORD.json.sig < PUBLICATION_RECORD.json
```

Compare the published archive checksum with `deposit_archive.sha256` in the signed record. See [evidence fields and run mapping](EVIDENCE.md) for what is inside. The reserved dataset DOI is [10.5281/zenodo.23989888](https://doi.org/10.5281/zenodo.23989888).

The companion OpenTimestamps receipt commits to the manifest digest through public calendars. The initial receipt is a calendar commitment. A Bitcoin attestation can be verified once the calendars have anchored it.

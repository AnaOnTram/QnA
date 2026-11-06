# Upload to GitHub and Hugging Face

This repository is prepared locally with an initial Git/LFS commit. No remote
repository or upload was created by packaging. Replace `OWNER` or `NAMESPACE`
below with the intended account or organization; use a new empty destination.

## Verify before upload

```bash
cd ~/QA/model-weights
export PATH="$(git rev-parse --absolute-git-dir)/tools/bin:$PATH"
sha256sum --check SHA256SUMS
git lfs fsck
git status --short
```

On another checkout use the installed Git LFS on PATH instead of this checkout's
private tool path. Full binary payloads must be present before uploading to the
Hub with `hf`; do not upload a checkout containing only LFS pointer text.

## GitHub: Git history and LFS binaries

Create an empty GitHub repository named `model-weights` in the intended account.
Leave its README/license/gitignore initialization disabled so the initial push
does not conflict with unrelated history. Then add the remote and push:

```bash
git remote add origin https://github.com/OWNER/model-weights.git
git push -u origin main
```

The LFS pre-push hook uploads the weight payloads. Check that all 24 files appear
as LFS files on GitHub, and retain `SHA256SUMS` and `manifest.json` alongside them.
GitHub credentials and LFS storage/bandwidth availability belong to the
destination account. Subsequent updates use `git push origin main` after the
documented weight-update workflow.

Reference: [GitHub LFS configuration and push behavior](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage).

## Hugging Face: upload the working files

Use the [current `hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
in a separate tool environment. It is not a dependency of the model archive.
The HTTP upload route handles the Hub's large-file storage without configuring
Git Xet or changing this checkout's LFS filters.

Authenticate, create a model repository, and upload an explicit allowlist:

```bash
hf auth login
hf repos create NAMESPACE/model-weights --repo-type model
hf upload NAMESPACE/model-weights . . --repo-type model \
  --include 'README.md' --include 'SHA256SUMS' --include 'manifest.json' \
  --include '.gitattributes' --include '.gitignore' \
  --include 'docs/**' --include 'weights/**' \
  --exclude '**/.cache/**' \
  --commit-message "Upload runtime model weights and documentation"
```

`README.md` is already a Hub model card with image-classification metadata. The
allowlist sends weights and documentation and excludes the local `.git` directory
and its tooling. Verify that the remote file inventory and checksums match the
local manifest. Repeat `hf upload` after each committed update; it uploads file
changes but does **not** mirror local Git commit IDs or history. A Hub upload also
does not automatically remove remotely retained files that were deleted locally;
review such removals separately.

This is a collection with subfolders for two architectures and multiple runtimes;
consumers should download the required files. It is not a single Transformers
model with one root `config.json`. Future publication can split the model families
into separate Hub repositories while preserving the per-file checksums.

Reference: [Hugging Face repository guide](https://huggingface.co/docs/hub/repositories-getting-started)
and [HTTP upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload).

## Tracking releases on both hosts

GitHub remains the Git/LFS history source in the workflow above. For a named
release, tag the local commit and publish the Git tag to GitHub:

```bash
git tag -a weights-v1 -m "Initial runtime weight snapshot"
git push origin weights-v1
```

Use the Git commit ID in the corresponding Hub upload message or description,
and record the returned Hub revision when documenting a release. Each host has
its own revision identifiers; `SHA256SUMS` establishes byte identity across them.

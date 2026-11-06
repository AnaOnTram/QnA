# Git tracking and weight updates

The repository uses `main`. All model binary extensions are matched by the
committed `.gitattributes` LFS rules before the initial commit. Model files in
Git history are small LFS pointers; complete payloads live in LFS storage and in
the working tree. Documentation, `manifest.json` and `SHA256SUMS` use regular Git.

## This local checkout

The original machine had no Git LFS on PATH. Git LFS **3.8.1** was downloaded
from its official GitHub release, checked against the release asset SHA-256, and
installed only at `.git/tools/bin/git-lfs`. The original checkout has local
filter paths and a pre-push hook pointing to this binary. No system installation
or global Git configuration was changed. This `.git` tooling is not committed
or uploaded.

For diagnostic `git lfs` commands in this checkout, add the private tool to the
current shell's PATH:

```bash
cd ~/QA/model-weights
export PATH="$(git rev-parse --absolute-git-dir)/tools/bin:$PATH"
git lfs version
git lfs ls-files
git lfs fsck
git status --short
sha256sum --check SHA256SUMS
```

The original checkout's filter configuration contains absolute paths. If you
move it, put an installed Git LFS on PATH and run `git lfs install --local --force`
to refresh the local filters and hook.

## Fresh clone

Install [Git LFS](https://git-lfs.com/) on the receiving machine. Clone without
automatically downloading binaries, then enable LFS locally and fetch them:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/OWNER/model-weights.git
cd model-weights
git lfs install --local
git lfs pull
sha256sum --check SHA256SUMS
```

For a smaller checkout, fetch one model family:

```bash
git lfs pull --include='weights/resnetv2_50/**' --exclude=''
```

Full `SHA256SUMS` verification requires all 24 weight files to be downloaded.
A text LFS pointer in place of a binary indicates its payload has not been fetched.

## Updating weights

Copy a replacement or new runtime variant into the corresponding model/runtime
folder. Keep the source outside this repository unchanged. Update its
`manifest.json` entry with the source, bytes, SHA-256, recipe and target metadata;
update `docs/CATALOG.md` and the README totals/matrix if the inventory changes.
Then regenerate and verify checksums from the current manifest:

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
p = Path('manifest.json')
m = json.loads(p.read_text())
for a in m['artifacts']:
    f = Path(a['path'])
    a['bytes'] = f.stat().st_size
    with f.open('rb') as stream:
        a['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
m['artifacts'].sort(key=lambda a: a['path'])
p.write_text(json.dumps(m, indent=2) + '\n')
Path('SHA256SUMS').write_text(''.join(
    a['sha256'] + '  ' + a['path'] + '\n' for a in m['artifacts']))
PY
sha256sum --check SHA256SUMS
git add .gitattributes .gitignore README.md docs manifest.json SHA256SUMS weights
git lfs status
git diff --cached --stat
git diff --cached -- docs README.md manifest.json SHA256SUMS
git commit -m "Update model weight artifacts and provenance"
```

The checksum-update snippet requires Python 3.11 or newer. For a rebuild, replace
source metadata as well as checksums: old source hashes and calibration records
must not be carried forward as if they described the new artifact.

Avoid committing full binary blobs with LFS filters disabled. Keep `.gitattributes`
in every distribution and retain the LFS pre-push hook. See
[GitHub's LFS configuration guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage).

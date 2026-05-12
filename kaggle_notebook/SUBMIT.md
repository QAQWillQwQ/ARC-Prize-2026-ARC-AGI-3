# Kaggle Submission Guide — ARC AGI 3 Warmstart v1

End-to-end workflow with the lessons learned during the first push (May 2026 session). Read **§ Hard-won lessons** first if you're hitting errors.

---

## Prereqs (one-time)

```bash
pip install kaggle

# 1. Get token from https://www.kaggle.com/settings → "Create New API Token"
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 2. Verify auth
kaggle competitions list | head             # should print competitions
python -c "import json; print('username:', json.load(open('$HOME/.kaggle/kaggle.json'))['username'])"
```

Note the printed `username:` — every `id` field in metadata files MUST start with this exact string (case-sensitive). Otherwise the API returns `Invalid Owner Id`.

---

## Step 1 — Edit metadata files

Replace `USERNAME` with your actual Kaggle handle in **all three** files:
- `kaggle_notebook/metadata/kernel-metadata.json` — `id` and `dataset_sources`
- `kaggle_notebook/metadata/dataset-metadata-replays.json` — `id`
- The staged copy at `Local_Output/kaggle_dataset_replays/dataset-metadata.json` (after step 2 below)

**The slug after `USERNAME/` must equal `slugify(title)`** (see § Hard-won lessons #1 below). Easiest: set both `id` and the title-derived slug to the same string.

---

## Step 2 — Build + push the replays dataset

```bash
# Stage a clean dir with only the replays (no game .py, no metadata.json)
mkdir -p ./Local_Output/kaggle_dataset_replays/environment_files
for game in environment_files/*/; do
    short=$(basename "$game")
    if [ -d "$game/replays" ]; then
        mkdir -p "./Local_Output/kaggle_dataset_replays/environment_files/$short"
        cp -r "$game/replays" "./Local_Output/kaggle_dataset_replays/environment_files/$short/"
    fi
done

# Copy dataset metadata
cp kaggle_notebook/metadata/dataset-metadata-replays.json \
   ./Local_Output/kaggle_dataset_replays/dataset-metadata.json
# (If you edited only the template after this point, copy again or also edit
#  the staged copy — both must agree.)

# v4.1: also stage the action-effect retrieval dict (loaded by my_agent.py
# at runtime from /kaggle/input/arc-agi-3-replays-v1/action_effect_dict.npz)
cp Local_Output/action_effect_dict.npz \
   ./Local_Output/kaggle_dataset_replays/action_effect_dict.npz

# v4.3: also stage the BC checkpoint (loaded by bc_policy.find_checkpoint at
# runtime; cell 3 of the notebook copies it to /kaggle/working/best.pth so
# MyAgent.__init__ can load it for TTT). Use whichever .pth file you want
# to ship as the base — the most recent local-trained one is recommended.
cp Training_Output/bc_v2_filtered_local_v1/checkpoints/best.pth \
   ./Local_Output/kaggle_dataset_replays/best.pth

# Push (--dir-mode zip is MANDATORY for the nested per-game directories)
kaggle datasets create -p ./Local_Output/kaggle_dataset_replays --dir-mode zip
```

After upload, browse to https://www.kaggle.com/datasets/jihangli1121/arc-agi-3-replays-v1 and **set visibility to Public** (per External Data rule §2.6 — easier compliance).

To update the dataset later:
```bash
kaggle datasets version -p ./Local_Output/kaggle_dataset_replays \
  -m "v2 replays" --dir-mode zip
```

**v4.1 dict bundle:** if the dataset is already pushed, add the npz with:
```bash
cp Local_Output/action_effect_dict.npz \
   ./Local_Output/kaggle_dataset_replays/action_effect_dict.npz
kaggle datasets version -p ./Local_Output/kaggle_dataset_replays \
  -m "v3 add action_effect_dict.npz" --dir-mode zip
```
No kernel-side change is needed — `my_agent.py:_load_action_effect_dict()`
already searches the canonical Kaggle mount path. Set
`ARC_DISABLE_EFFECT_DICT=1` in the env for local A/B testing.

**v4.3 BC checkpoint bundle:** to enable TTT (test-time training), add the
.pth file to the same dataset:
```bash
cp Training_Output/bc_v2_filtered_local_v1/checkpoints/best.pth \
   ./Local_Output/kaggle_dataset_replays/best.pth
kaggle datasets version -p ./Local_Output/kaggle_dataset_replays \
  -m "v4 add BC checkpoint best.pth (~190 MB)" --dir-mode zip
```
Notebook cell `c-stage` (cell 3) automatically copies the .pth from the
mounted dataset to `/kaggle/working/best.pth` at submission time.
`bc_policy.find_checkpoint()` then locates it. Disable TTT via
`ARC_DISABLE_TTT=1` for clean A/B. The BC checkpoint can be swapped at any
time without changing notebook code — only the dataset version bumps.

---

## Step 3 — Push the notebook

```bash
mkdir -p ./Local_Output/kaggle_kernel
cp kaggle_notebook/notebooks/kaggle_submission.ipynb ./Local_Output/kaggle_kernel/
cp kaggle_notebook/metadata/kernel-metadata.json     ./Local_Output/kaggle_kernel/

kaggle kernels push -p ./Local_Output/kaggle_kernel
```

The push triggers a "Save & Run All" on Kaggle. The dev-mode run executes (cells 1-4 + cell 6, since `KAGGLE_IS_COMPETITION_RERUN` is unset) and writes a dummy `submission.parquet`.

Watch status:
```bash
kaggle kernels status jihangli1121/arc-agi-3-warmstart-v1
```

Wait for `KernelWorkerStatus.COMPLETE` before submitting. If status reads `error`, open the kernel URL in browser → Logs tab to see the traceback.

---

## Step 4 — Submit to competition

**Strongly recommended: use the web UI**, not the CLI. Code competitions on Kaggle reliably accept submissions via the **"Submit to Competition"** button on the kernel page; the CLI submission API (`kaggle competitions submit -k ... -v ...`) frequently returns `400 Bad Request` for code competitions even when everything else is correct.

Web UI flow:
1. Open https://www.kaggle.com/code/jihangli1121/arc-agi-3-warmstart-v1
2. Top-right (or under the version dropdown): **"Submit to Competition"**
3. Pick the version (typically the latest) → confirm
4. Kaggle queues the rerun against the gateway. Track from the competition's "My Submissions" tab.

If you must use the CLI, try these in order:
```bash
# Variant 1 (positional competition):
kaggle competitions submit arc-prize-2026-arc-agi-3 \
  -k jihangli1121/arc-agi-3-warmstart-v1 -v 1 -m "v1 warmstart"

# Variant 2 (explicit -c flag):
kaggle competitions submit -c arc-prize-2026-arc-agi-3 \
  -k jihangli1121/arc-agi-3-warmstart-v1 -v 1 -m "v1 warmstart"

# Variant 3 (no version):
kaggle competitions submit -c arc-prize-2026-arc-agi-3 \
  -k jihangli1121/arc-agi-3-warmstart-v1 -m "v1 warmstart"
```

---

## What happens during the competition rerun

When you submit, Kaggle:
1. Spins up a fresh container with `KAGGLE_IS_COMPETITION_RERUN=1` set
2. Mounts your dataset(s) at `/kaggle/input/<dataset-slug>/`
3. Mounts the competition's framework at `/kaggle/input/competitions/arc-prize-2026-arc-agi-3/`
4. Starts the gateway service at `http://gateway:8001/`
5. Re-runs the notebook end-to-end. Cell 5 fires (since `KAGGLE_IS_COMPETITION_RERUN` is set), copies the framework, drops `MyAgent` into `agents/templates/`, writes `.env`, runs `python main.py --agent myagent`
6. The framework iterates all gateway games, calls `MyAgent.choose_action()` per step, posts the score to the leaderboard

The rerun is much slower than the dev run — minutes per game × 25 games × however many resets the agent burns. Plan for 30min+ wall time on the rerun.

---

## Iterating after the first submission

After a code change:

```bash
# Re-stage the notebook
cp kaggle_notebook/notebooks/kaggle_submission.ipynb ./Local_Output/kaggle_kernel/

# Push (creates a new version automatically)
kaggle kernels push -p ./Local_Output/kaggle_kernel
kaggle kernels status jihangli1121/arc-agi-3-warmstart-v1   # wait for COMPLETE

# Submit the new version (web UI button, OR CLI):
kaggle competitions submit arc-prize-2026-arc-agi-3 \
  -k jihangli1121/arc-agi-3-warmstart-v1 -v 2 -m "v2 desc"
```

If you also changed replays:
```bash
kaggle datasets version -p ./Local_Output/kaggle_dataset_replays \
  -m "v2 replays" --dir-mode zip
```

Datasets are versioned independently from kernels — you don't need to bump the kernel version when only data changes (Kaggle re-resolves dataset_sources to the latest version on each kernel run).

---

## Daily limits & strategy

- **5 submissions per day** max (rules §2.2). Don't burn 5 on day one — failed submissions still count.
- **2 Final picks** at competition close. Iterate freely on Public LB during development; only the 2 you select are scored on Private LB.
- **Public LB ≠ Private LB**. Warmstart's strength (replay-on-public-games) does NOT carry to Private. Public/milestones reward warmstart; Private rewards generalization (= the worldmodel_v4 retrain on `replays_v2.gz` covered in `.claude/doc/worldmodel_v3.5_plan.md`).
- **Milestone 1: 2026-06-30** — best leaderboard score on that date wins. Keep at least one valid submission live.

---

## Hard-won lessons

### 1. Kernel slug = slugify(title), not your `id` field

If `kernel-metadata.json` has `id: "user/foo"` but `title: "ARC AGI 3 Warmstart v1"`, Kaggle creates the kernel at slug `arc-agi-3-warmstart-v1` (slugified from title) — **not** at `foo`. The push prints `Your kernel title does not resolve to the specified id` as a warning that's easy to miss. Status queries against the wrong slug return:

```
Cannot access kernel '<owner>/<bad-slug>' (Permission 'kernels.get' was denied).
```

**Fix**: make the `id` slug match `slugify(title)`. Either change the title or the id so they agree. After Kaggle creates a kernel at a particular slug, the slug is permanent for that kernel — update the local `id` to match what Kaggle actually used.

### 2. "Invalid Owner Id" on `kaggle datasets create`

The `id` in `dataset-metadata.json` must start with your authenticated Kaggle username (case-sensitive). Common mistakes:
- Forgot to replace `USERNAME` placeholder
- Edited the template at `kaggle_notebook/metadata/dataset-metadata-replays.json` but not the staged copy at `Local_Output/kaggle_dataset_replays/dataset-metadata.json`. The CLI reads from the staged folder.
- Different casing between the file and `~/.kaggle/kaggle.json`

**Fix**:
```bash
python -c "import json; print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])"
# That's your authoritative username. Make every metadata id start with it.
grep '"id"' kaggle_notebook/metadata/dataset-metadata-replays.json \
            Local_Output/kaggle_dataset_replays/dataset-metadata.json
```

### 3. CLI `competitions submit` returns 400 for code competitions

Even with valid metadata, kernel slug, version number, and a successful run, the CLI submit endpoint frequently rejects code competition submissions with `400 Bad Request for url: ...CreateCodeSubmission`. This appears to be a Kaggle-side limitation — code competitions are designed around the web UI flow.

**Fix**: use the **"Submit to Competition"** button on the kernel page (https://www.kaggle.com/code/<owner>/<slug>). This is the canonical, reliable path for code competitions.

### 4. `--dir-mode zip` is mandatory for nested datasets

`environment_files/<game>/replays/*.json` has 25 nested directories. The default `--dir-mode skip` silently drops subdirectories during upload. Always pass `--dir-mode zip` for the replays dataset.

### 5. Internet flag for ARC AGI 3

Per the ARC Prize policy page, internet IS allowed for this competition. The samples (`arc3-sample-submission-random-agent.ipynb`, `arc3-sample-submission-stochastic-goose.ipynb`) don't set `enable_internet` explicitly. Our `kernel-metadata.json` sets it to `"false"` to match the safer default. If you ever need to fetch external models or APIs at rerun time, change it to `"true"` and re-push.

The `gateway:8001` service is internal Kaggle networking, so the gateway works regardless of the `enable_internet` setting.

### 6. `kernel-metadata.json` boolean fields are stringified

```json
{
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "false"
}
```

Bare booleans (`true`/`false` without quotes) silently get rejected. Always use the JSON-string form.

### 7. Both metadata files must stay in sync

After fixing the username in the template, **also update the staged copy** at `Local_Output/kaggle_dataset_replays/dataset-metadata.json`. The CLI does NOT re-read from `kaggle_notebook/`. Same for `kernel-metadata.json` (`Local_Output/kaggle_kernel/kernel-metadata.json` is the one Kaggle reads at push time).

---

## Diagnostic recipes

```bash
# Check what the kernel produced
kaggle kernels output jihangli1121/arc-agi-3-warmstart-v1 -p /tmp/kg_out
ls /tmp/kg_out/                 # should include submission.parquet
cat /tmp/kg_out/__results__.html 2>/dev/null | head -50  # the rendered HTML log

# Check submission history
kaggle competitions submissions arc-prize-2026-arc-agi-3

# Check kernel metadata as Kaggle sees it
kaggle kernels list -m --user jihangli1121 | grep warmstart

# Pull dataset back down to verify upload structure
kaggle datasets download -d jihangli1121/arc-agi-3-replays-v1 -p /tmp/ds
unzip -l /tmp/ds/arc-agi-3-replays-v1.zip | head -30
```

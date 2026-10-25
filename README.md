# SpyMotion

Patient-owned, on-device movement progress tracking for people recovering from spinal cord injury.
Film an exercise, get peak range of motion per repetition, consistency, and a before/after comparison
with the uncertainty stated. No cloud, no accounts, no API keys.

## Run it

### AccessScan

After setup, run this command from the repository root in Windows PowerShell:

```powershell
.\scripts\start_app.ps1

2.
```

Open http://127.0.0.1:8501 and select **AccessScan**:

1. **Wheelchair width (cm)**: the widest part of the wheelchair.
2. **Wheelchair height (cm)**: from the ground to the armrest.
3. Keep your phone upright and position the camera lens at armrest height. Include both passage edges and the ground.
4. Upload the photo and press **Analyse accessibility**.

The result is shown as a large **Can pass!**, **Cannot pass**, or **Unable to determine**.
The calculation requires camera-lens height, so the camera must be at the height entered above.
To reproduce `models/SidewalkDetection/images/image8.jpg`, use its known camera height of **124 cm**
in the height field and **65 cm** for wheelchair width.

First-time Windows setup, from the repository root (Python 3.10–3.12):

```powershell
git lfs pull
python -m venv models/SidewalkDetection/.venv
& .\models\SidewalkDetection\.venv\Scripts\python.exe -m pip install -r .\models\SidewalkDetection\requirements.txt
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\scripts\start_app.ps1
```

The frontend and model dependencies use separate environments. The model interpreter is found
in `models/SidewalkDetection/.venv`; `SIDEWALK_PYTHON` or a Git-ignored `accessscan.local.json` can override it.
Models stay loaded and the most recent photo's boundaries and calibration are reused when measurements change.
New photos still require model inference. Diagnostic results overwrite `models/SidewalkDetection/output`.

AccessScan uses pretrained **Meta SAM 2.1 Small** and **GeoCalib** without retraining.
Model details and licenses are retained in [models/SidewalkDetection/README_EN.md](models/SidewalkDetection/README_EN.md).
The width is the median across the selected visible region; it is not a verified minimum clearance or obstacle check.

### Movement analysis

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.9-3.12 (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
```

`mediapipe==0.10.14` is pinned on purpose (newer releases removed the bundled `mp.solutions` model), and
`pyarrow<19` is pinned because newer versions need numpy 2, which that mediapipe cannot use.
The first run needs internet only for `pip install`; afterwards the app works with Wi-Fi off.
Only the standard pose model ships inside the package; the "lite" and "heavy" models would download from
Google at first use, so the app never selects them.

## What it does

1. Pose estimation (MediaPipe) -> joint angle per frame, computed in pixel space.
2. Confidence gating: frames with low landmark visibility are dropped, short gaps bridged, long gaps flagged.
3. Spike removal + Savitzky-Golay smoothing, then **peak detection per repetition** (not a video-wide average).
4. Metrics: average peak angle, rep-to-rep spread, rep count, smoothness (movement units), tracking confidence.
5. **Refuses to score** if tracking coverage < 60% or < 2 reps; warns if the camera is not side-on or if a
   stretch of the video was lost (repetitions in it are not counted, and the user is told).
6. Before/after: Welch t-interval over rep peaks; a change is only called real if it exceeds the noise **and** 5 degrees.
7. Tips from fixed rules, each labelled with its rule ID (R1...R8). No generative model.
8. Result video: skeleton overlay, live angle, rep counter, **face pixelated, audio stripped**.
9. Optional daily wellbeing diary; the History tab plots movement next to pain/fatigue on days with both.
   The medication & routines tab is optional and **off by default** (sidebar toggle).

## What is stored on disk (and how to wipe it)

| What             | Where                                   | When                                              |
| ---------------- | --------------------------------------- | ------------------------------------------------- |
| Original video   | temp folder, only while analysing       | removed immediately, even if the analysis crashes |
| Result video     | memory only                             | its file is removed as soon as it is read         |
| Movement numbers | `spymotion_data/history.json`           | only if "Save my numbers" is on                   |
| Wellbeing diary  | `spymotion_data/wellbeing_entries.json` | only when the user presses Save                   |

Folders are owner-only; writes are atomic; a corrupt file is moved aside, never overwritten.
**Delete everything now** (sidebar) removes all of it and verifies it. The _Data & safety_ tab lists every file
currently held, so the claim can be checked live. Health data and videos are in `.gitignore`: never commit them.

## Testing & reliability

```bash
pip install -r requirements-dev.txt
pytest                         # 102 tests, ~30 s, no camera needed
python -m spymotion.scenarios           # reproducible regression table of what each synthetic case must produce
python scripts/check_clean_install.py  # builds a brand-new venv, installs, runs the whole suite
```

Covered: empty/corrupt/truncated/tiny/odd-size/odd-fps/short videos; MediaPipe losing the person (start, middle,
end, flicker); wrong side / front-on camera; storage corruption, disk-full, permissions; deletion and audit;
leftover-footage sweep; every UI results state; failed videos in the UI; the diary migration from older builds.

## Evidence tools (needs your clips)

```bash
python validate.py --csv data/labels.csv --videos clips/     # accuracy + per-condition fairness table -> validation_report.md
python tests/stress_test.py --video clips/good.mp4 --exercise knee # dim/blur/noise/crop/mirror from ONE clip -> stress_report.md
```

Both reports are shown automatically in the _Data & safety_ tab once they exist.

## Known limitations (say these out loud in the pitch)

- 2D projection: needs a side-on, fixed camera. Smoothing slightly attenuates peaks (synthetic test: ~1-2%).
- Pose models are trained mostly on able-bodied people; wheelchair/assisted postures can fail, hence the refusal rule.
- Face blur is best-effort. History and diary files are not encrypted. Typical-range values are textbook averages.
- Not a medical device; the diary is not a validated clinical PROM; tips would need clinician review before real use.
- Tested on Linux / Python 3.12 only. Windows and macOS are untested.

## Files

app.py — Streamlit UI

spymotion/

- core.py — movement-analysis engine
- pose_pipeline.py — video processing and MediaPipe
- robust.py — crash containment and reliability
- store.py — local storage and deletion
- insights.py — progress/history insights
- scenarios.py — synthetic regression scenarios
- accessscan/ — AccessScan application logic

models/

- SidewalkDetection/ — pretrained accessibility-detection models

tests/

- test\_\*.py — automated tests
- conftest.py — pytest configuration
- stress_test.py — robustness/stress testing

scripts/

- start_app.ps1 — Windows launcher
- check_clean_install.py — clean-install verification

data/

- labels.csv — validation labels
- labels_template.csv — validation template

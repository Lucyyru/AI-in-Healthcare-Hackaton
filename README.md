# SpineMotion

Patient-owned, on-device movement progress tracking for people recovering from spinal cord injury.
Film an exercise, get peak range of motion per repetition, consistency, and a before/after comparison
with the uncertainty stated. No cloud, no accounts, no API keys.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.9-3.12
pip install -r requirements.txt
python test_core.py                                      # 9 tests, no camera needed
streamlit run app.py
```
`mediapipe==0.10.14` is pinned on purpose (newer releases removed the bundled `mp.solutions` model).
First run needs internet only for `pip install`; afterwards the app works with Wi-Fi off.

## What it does
1. Pose estimation (MediaPipe) -> joint angle per frame, computed in pixel space.
2. Confidence gating: frames with low landmark visibility are dropped, short gaps bridged, long gaps flagged.
3. Spike removal + Savitzky-Golay smoothing, then **peak detection per repetition** (not a video-wide average).
4. Metrics: average peak angle, rep-to-rep spread, rep count, smoothness (movement units), tracking confidence.
5. **Refuses to score** if tracking coverage < 60% or < 2 reps; warns if the camera is not side-on.
6. Before/after: Welch t-interval over rep peaks; a change is only called real if it exceeds the noise **and** 5 degrees.
7. Tips from fixed rules, each labelled with its rule ID (R1...R8). No generative model.
8. Result video: skeleton overlay, live angle, rep counter, **face pixelated, audio stripped**. Original upload is deleted after analysis.

## Your job: the data (this is where the fairness prize is won)
Film **consented** clips of yourselves (and teammates), side-on, 3-5 slow reps, 15-30 s.

| Set | Why |
|---|---|
| Same exercise, 3 range levels (full / reduced / very limited) | Shows "improved / no change / lower" in the demo |
| Standing **and seated** | Seated is the realistic SCI posture |
| **Wheelchair** (a teammate sits in one if you can borrow one) | Where pose models are weakest |
| Dim light, dark clothing, loose clothing, cluttered background | Fairness-by-condition table |
| A deliberately bad clip (front-on camera, body cropped) | Proves the refusal logic live |

**Reference measurement:** for each clip, pause on the frame of maximum bend and measure the angle (phone
protractor/goniometer app on the screenshot, or a real goniometer held at the hold position). Put it in
`labels.csv` (see `labels_template.csv`), then:
```bash
python validate.py --csv labels.csv --videos clips/
```
This writes `validation_report.md`, which the app shows in the **Data & safety** tab automatically.
Be honest about what it validates: pose estimate vs your manual reading, not a clinical trial.

## Known limitations (say these out loud in the pitch)
- 2D projection: needs a side-on, fixed camera. Rep peaks are ~1-2% attenuated by smoothing (synthetic test: 83.3 vs 85).
- Pose models are trained mostly on able-bodied people; wheelchair/assisted postures can fail, hence the refusal rule.
- Face blur is best-effort. History file is not encrypted. Typical-range values are textbook averages.
- Not a medical device; tips would need clinician review before real use.

## Pitch map (4 min)
1. **Story 30s**: patient exercising at home, no idea if she's improving.
2. **Demo 90s**: upload Earlier + Latest -> headline change with +/- range -> skeleton videos side by side -> tips with rule IDs.
   Then upload the bad clip to show it refuses. Switch Wi-Fi off before the demo.
3. **How it works 60s**: per-rep peaks, gating, uncertainty-aware comparison.
4. **Responsible data 45s**: open the Data & safety tab: live file audit, delete button, fairness table.
5. **Close 15s**: patient agency.

## Likely Q&A
- *How accurate vs clinical measurement?* Show validation_report.md; state 2D limits and side-on requirement.
- *Wheelchair users?* Tested; show the condition table, and the refusal behaviour where it fails.
- *Medical device?* No: self-monitoring aid; safety notice in-app; clinician review needed before deployment.
- *Why no LLM?* Deterministic, auditable tips; nothing to hallucinate; nothing leaves the device.

## Files
`app.py` UI - `core.py` engine - `pose_pipeline.py` video/MediaPipe - `store.py` storage + delete - `validate.py` evidence - `test_core.py` tests

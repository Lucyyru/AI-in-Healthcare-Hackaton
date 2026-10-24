# SpineMotion

Patient-owned, on-device movement progress tracking for people recovering from spinal cord injury.
Film an exercise, get peak range of motion per repetition, consistency, and a before/after comparison
with the uncertainty stated.
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

## Files
`app.py` UI - `core.py` engine - `pose_pipeline.py` video/MediaPipe - `store.py` storage + delete - `validate.py` evidence - `test_core.py` tests

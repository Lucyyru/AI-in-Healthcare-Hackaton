# SpyMotion

Patient-owned, on-device movement progress tracking for people recovering from spinal cord injury.

Film an exercise, measure peak range of motion per repetition, assess consistency, and compare progress over time with uncertainty made explicit. SpyMotion is designed to run locally: no accounts, API keys, or cloud upload are required.

## Run it

### How to open

#### Windows PowerShell

Make sure you first do :
cd ".../main"
... = path to the folder "SpyMotion-main"

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

### AccessScan

AccessScan estimates whether a wheelchair can fit through a photographed doorway or passage.

After completing the setup below, run this command from the repository root in Windows PowerShell:

```powershell
.\scripts\start_app.ps1
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501) and select **AccessScan**.

1. Enter **Wheelchair width (cm)** — measure the widest part of the wheelchair.
2. Enter **Wheelchair height (cm)** — measure from the ground to the armrest.
3. Keep the phone upright and position the camera lens at approximately the entered armrest height. Include both passage edges and the ground.
4. Upload the photo and press **Analyse accessibility**.

The result is displayed as **Can pass!**, **Cannot pass**, or **Unable to determine**.

The calculation requires camera-lens height, so the camera should be positioned at approximately the height entered in the app.

To reproduce the included example at `models/SidewalkDetection/images/image8.jpg`, use:

- Camera height: **124 cm**
- Wheelchair width: **65 cm**

### First-time Windows setup

Run the following from the repository root using Python 3.10–3.12:

```powershell
git lfs pull

python -m venv models/SidewalkDetection/.venv
& .\models\SidewalkDetection\.venv\Scripts\python.exe -m pip install -r .\models\SidewalkDetection\requirements.txt

python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r .\requirements.txt

.\scripts\start_app.ps1
```

The frontend and AccessScan model dependencies use separate Python environments.

By default, the AccessScan model interpreter is expected at:

```text
models/SidewalkDetection/.venv
```

`SIDEWALK_PYTHON` or a Git-ignored `accessscan.local.json` file can override the interpreter location.

Models remain loaded while the application is running, and the most recent image boundaries and calibration can be reused when measurements change. A new image still requires model inference.

Diagnostic output is written under:

```text
models/SidewalkDetection/output
```

AccessScan uses pretrained **Meta SAM 2.1 Small** and **GeoCalib** models without retraining.

Model details and licence information are retained in:

[models/SidewalkDetection/README_EN.md](models/SidewalkDetection/README_EN.md)

The estimated width is based on the selected visible region. It is not a certified minimum-clearance measurement or an obstacle-detection system.

---

### Movement analysis

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

`mediapipe==0.10.14` is pinned intentionally because later releases changed availability of the bundled `mp.solutions` interface used by the project.

The first setup requires internet access to install dependencies. The movement-analysis pipeline itself is designed to operate locally after installation.

## Interface

SpyMotion is organised around four main areas:

- **Analyse Movement** — record or upload rehabilitation movements and receive objective measurements.
- **AccessScan** — estimate whether a wheelchair can fit through a photographed passage.
- **My Recovery** — view movement history and optional daily wellbeing context.
- **More** — access medication and routines, data/privacy controls, and settings.

The sidebar also provides:

- **Phone / Website view**
- **Dark mode**

## What it does

1. **Pose estimation**  
   MediaPipe estimates body landmarks and calculates joint angles frame by frame.

2. **Confidence gating**  
   Frames with insufficient landmark visibility are discarded. Short gaps may be bridged while longer unreliable sections remain excluded.

3. **Signal processing and repetition detection**  
   The joint-angle signal is cleaned and smoothed before detecting individual repetition peaks.

4. **Movement metrics**  
   SpyMotion reports metrics including:
   - Average peak angle
   - Repetition count
   - Rep-to-rep variation
   - Smoothness / movement units
   - Tracking coverage
   - Tracking confidence

5. **Reliability checks**  
   SpyMotion refuses to produce a movement score when tracking is not sufficiently reliable rather than presenting an uncertain measurement as fact.

6. **Progress comparison**  
   Saved sessions can be compared over time to identify meaningful changes while accounting for measurement variability.

7. **Rule-based feedback**  
   Patient-facing tips are produced using deterministic rules rather than a generative model.

8. **Analysed result video**  
   The result can include:
   - Skeleton overlay
   - Live joint angle
   - Repetition counter
   - Face pixelation
   - Removed audio

9. **Daily wellbeing tracking**  
   Users can optionally record wellbeing information and view it alongside rehabilitation progress.

10. **Medication & routines**  
    Supporting medication and routine tracking is available under **More → Medication & routines**.

## Privacy and local storage

SpyMotion is designed around local processing and explicit saving.

| Data             | Location                                | When stored                                                   |
| ---------------- | --------------------------------------- | ------------------------------------------------------------- |
| Original video   | Temporary processing location           | Removed after analysis                                        |
| Result video     | Temporary / session processing          | Removed after it is consumed                                  |
| Movement history | `spymotion_data/history.json`           | Only when the user explicitly saves a result in Progress mode |
| Wellbeing diary  | `spymotion_data/wellbeing_entries.json` | Only when the user saves a wellbeing entry                    |

Storage writes are handled locally. Corrupt data files are moved aside rather than silently overwritten.

To remove saved application data, open:

**More → Settings → Delete everything now**

The **More → Data & privacy** section shows information about locally stored data so the application's privacy claims can be inspected during the demo.

Health-related local data and video files should remain excluded from Git through `.gitignore`.

## Testing and reliability

Install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the complete automated test suite:

```bash
python -m pytest tests
```

Run the synthetic movement scenarios:

```bash
python -m spymotion.scenarios
```

Run the clean-install verification:

```bash
python scripts/check_clean_install.py
```

The automated tests cover areas including:

- Empty and corrupt input
- Truncated and very short videos
- Unusual video sizes and frame rates
- Tracking loss at different points in a recording
- Low-confidence pose estimates
- Incorrect camera orientation
- Storage corruption
- Data deletion and audit behaviour
- Temporary-file cleanup
- UI result states
- Failed-video handling
- Wellbeing-data migration
- AccessScan behaviour

The current project has also been tested locally on **Windows with Python 3.10**.

## Evidence tools

These tools require your own validation clips.

### Accuracy and fairness validation

```bash
python validate.py --csv data/labels.csv --videos clips/
```

This produces:

```text
validation_report.csv
validation_report.md
```

The validation report includes accuracy measurements, refusal/failure counts, and results grouped by recorded condition.

### Stress testing

```bash
python tests/stress_test.py --video clips/good.mp4 --exercise knee
```

The stress test generates altered versions of a reference clip to evaluate robustness to conditions such as blur, noise, cropping, mirroring, and reduced visibility.

Generated evidence reports can be surfaced through the application's **Data & privacy** interface when available.

## Known limitations

- SpyMotion uses **2D pose estimation**, so movement measurements depend on camera position and viewing angle.
- A fixed, approximately side-on camera is recommended for movement analysis.
- Smoothing can slightly attenuate detected movement peaks.
- Pose-estimation models may perform differently across body types, mobility aids, assisted movements, clothing, lighting conditions, and camera angles.
- SpyMotion therefore includes explicit reliability checks and can refuse to score recordings rather than guessing.
- AccessScan provides an estimated visible passage width and is not a certified accessibility or minimum-clearance measurement.
- Face pixelation is best-effort.
- Saved history and wellbeing files are local but are not encrypted.
- Reference movement ranges are informational rather than patient-specific clinical targets.
- The wellbeing diary is not a validated clinical PROM.
- Rule-based tips would require clinical review before use in a real healthcare setting.
- SpyMotion is a prototype and **not a medical device**.

## Project structure

```text
SpyMotion/
├── app.py
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .gitignore
├── .gitattributes
│
├── spymotion/
│   ├── __init__.py
│   ├── core.py
│   ├── pose_pipeline.py
│   ├── robust.py
│   ├── store.py
│   ├── insights.py
│   ├── scenarios.py
│   └── accessscan/
│       ├── __init__.py
│       ├── accessscan.py
│       └── accessscan_worker.py
│
├── models/
│   └── SidewalkDetection/
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── stress_test.py
│   └── test_*.py
│
├── scripts/
│   ├── start_app.ps1
│   └── check_clean_install.py
│
├── data/
│   ├── labels.csv
│   └── labels_template.csv
│
├── videos/
│
└── validate.py
```

### Important files

- `app.py` — Streamlit user interface
- `spymotion/core.py` — movement-analysis engine
- `spymotion/pose_pipeline.py` — video processing and MediaPipe integration
- `spymotion/robust.py` — reliability and failure handling
- `spymotion/store.py` — local storage, audit, and deletion
- `spymotion/insights.py` — longitudinal progress and history insights
- `spymotion/scenarios.py` — reproducible synthetic regression scenarios
- `spymotion/accessscan/` — AccessScan application logic
- `models/SidewalkDetection/` — pretrained accessibility-analysis models and dependencies
- `tests/` — automated tests and stress-testing utilities
- `scripts/start_app.ps1` — Windows launcher
- `scripts/check_clean_install.py` — clean-install verification
- `data/` — validation labels and templates
- `validate.py` — accuracy and fairness validation utility

## Responsible-use principle

SpyMotion's core principle is:

> **Measure when the signal is reliable. Refuse when it is not.**

The system is intended to make rehabilitation progress easier to observe while keeping uncertainty, limitations, and user control visible.

## Credit
Lucia Rugeroni - Front end implementation, pose estimation, movement analysis, medication reminder.

Emanuele Marangoni - Front end implementation, pose estimation, movement analysis, medication reminder.

Qingteng Zhang - Access Scan implementation, model integration, perspective and depth calculation.

Junjie Zheng - Access Scan implementation, model integration, perspective and depth calculation.

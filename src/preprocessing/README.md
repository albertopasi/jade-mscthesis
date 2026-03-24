# Preprocessing

Converts raw EEG downloads to standardised `.npy` arrays ready for model input.

Both pipelines: resample 250 Hz → 200 Hz using `scipy.signal.resample` (FFT-based, matching the official REVE preprocessing). Values are saved as raw µV float32 — the `/1000` scale factor is applied at dataset load time, not here.

---

## FACED

**Input:** `data/FACED/Processed_data/sub{NNN}.pkl` — shape `(28, 32, 7500)` @ 250 Hz

**Output:** `data/FACED/preprocessed_v2/sub{NNN}.npy` — shape `(28, 32, 6000)` @ 200 Hz

```bash
# Process all 123 subjects
uv run python -m src.preprocessing.faced.run_preprocessing

# Process specific subjects
uv run python -m src.preprocessing.faced.run_preprocessing --subjects 0 1 2

# Validate output shapes and check for NaN/Inf
uv run python -m src.preprocessing.faced.run_preprocessing --validate
```

---

## THU-EP

**Input:** `data/thu ep/EEG data/sub_XX.mat` — HDF5 mat files, shape `(7500, 32, 28, 6)` @ 250 Hz, 6 frequency bands

**Output:** `data/thu ep/preprocessed_v2/sub_XX.npy` — shape `(28, 30, 6000)` @ 200 Hz

Steps: extract broadband (band index 5) → remove A1/A2 electrodes → resample 7500→6000

```bash
# Process all subjects
uv run python -m src.preprocessing.thu_ep.run_preprocessing

# Process specific subjects
uv run python -m src.preprocessing.thu_ep.run_preprocessing --subjects 1 2 3

# Validate preprocessed data
uv run python -m src.preprocessing.thu_ep.run_preprocessing --validate

# Dry run (list subjects without processing)
uv run python -m src.preprocessing.thu_ep.run_preprocessing --dry-run
```

---

## Notes

- Subject 75 (THU-EP) is corrupted and will be skipped automatically.
- Subject 37 (THU-EP): stimuli {15, 21, 24} excluded at dataset load time.
- Subject 46 (THU-EP): stimuli {3, 9, 17, 23, 26} excluded at dataset load time.
- FACED has no excluded subjects.
- Preprocessing must be run before any training (`train_lp.py`).

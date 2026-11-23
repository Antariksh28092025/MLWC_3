# COM 837 Assignment 3 - ChannelNet for OFDM channel estimation

SISO-OFDM, 72 subcarriers x 14 OFDM symbols, VehA and SUI5 channels, lattice pilots (16/24/32/48).
Estimators: LS + 2-D interpolation, MMSE (known statistics), ChannelNet (SRCNN -> DnCNN).

## 1. Files

| File | Purpose |
|---|---|
| `run_all.m` | MATLAB, one file: channel generation (simulator), pilots, noisy LS estimates, interpolation matrices, LS baseline, CSV tables. Helper functions are at the bottom of the file |
| `data_utils.py` | Reads the MATLAB files, builds network inputs/targets, normalisation |
| `models.py` | SRCNN and DnCNN |
| `estimators.py` | MMSE estimator (correlation matrices from the training channels) |
| `check_step7.py` | Checks that the Python pipeline reproduces the MATLAB baseline MSE |
| `train.py` | Trains one model (SRCNN stage, then DnCNN stage) for one channel / SNR / pilot count |
| `test.py` | Evaluates all methods; makes Figs. 3, 4, 5 and the tables |

## 2. Environment

- MATLAB (R2019b or newer; uses `scatteredInterpolant`, `matfile`, `writetable`) and the 5G link-level simulator supplied with the assignment (the folder that directly contains `+Modulation` and `+Channel`).
- Python 3.10+ with: `pip install numpy scipy h5py torch matplotlib python-docx`
- A GPU is optional. The runs reported here were done on a CPU with reduced settings (see section 5).

## 3. Exact execution steps

1. **Dataset (MATLAB).** Open `run_all.m`, set `simRoot` (simulator folder) and `dataDir` (output folder, outside OneDrive), press Run.
   It writes `H_VehA.mat`, `H_SUI5.mat` (ground truth, 32000/4000/4000 split), `LS_<chan>_P<pilots>_snr<dB>.mat`, `W_*.mat` (interpolation matrices) and `baseline_LS_interp.mat`, and prints the sanity checks. Finished stages are skipped on a re-run.
2. **Point Python at the data.** Edit `DATA` in `data_utils.py`, or set the environment variable `A3_DATA` to `dataDir`.
3. **Check the pipeline.** `python check_step7.py` must print `A) PASS`, `B) PASS` and `round trip exact: True`.
4. **Train** (settings used for the reported results; run one at a time):
   ```
   python train.py --chan VehA --snr 12 --max_train 4000 --max_eval 1000 --epochs_sr 5 --epochs_dn 3 --dn_depth 8
   python train.py --chan VehA --snr 22 --max_train 4000 --max_eval 1000 --epochs_sr 5 --epochs_dn 3 --dn_depth 8
   python train.py --chan SUI5 --snr 12 --max_train 4000 --max_eval 1000 --epochs_sr 5 --epochs_dn 3 --dn_depth 8
   python train.py --chan SUI5 --snr 22 --max_train 4000 --max_eval 1000 --epochs_sr 5 --epochs_dn 3 --dn_depth 8
   python train.py --chan VehA --snr 22 --pilots 24 --max_train 4000 --max_eval 1000 --epochs_sr 5 --epochs_dn 3 --dn_depth 8
   python train.py --chan VehA --snr 22 --pilots 32 --max_train 4000 --max_eval 1000 --epochs_sr 5 --epochs_dn 3 --dn_depth 8
   ```
   Paper-like setting (GPU recommended): omit `--max_train`, `--max_eval`, `--dn_depth` and use e.g. `--epochs_sr 20 --epochs_dn 15`.
   Outputs go to `checkpoints/`: `<run>_srcnn.pt`, `<run>_dncnn.pt`, `<run>_curves.png`, `<run>_curves.json`.
5. **Test and figures.** `python test.py` (add `--n_test 4000` for the whole test split). Outputs go to `results/`:
   Fig. 3 (VehA), Fig. 4 (SUI5), high-SNR figures, Fig. 5 (MSE vs pilots), CSV tables and `results.json`.
6. **Report.** `python make_report.py`, then open `Report_Assignment3.docx`, add the roll number and save as PDF.

## 4. Reproducibility

| Item | Value |
|---|---|
| Channel seeds | 42 (VehA), 43 (SUI5) |
| Pilot symbols | fixed QPSK sequence, seed 2024 |
| Noise seed | derived from channel, pilot spacing and SNR (stored in each `LS_*.mat`) |
| Training seed | 0 (`--seed`, stored in every checkpoint) |
| Channel settings | 15 kHz spacing, FFT 128, fs 1.92 MHz, CP 9 samples, 2.1 GHz, 50 km/h (f_d about 97.3 Hz), Jakes with 200 paths, channel read once per OFDM symbol (time-subsampling 137) |
| SNR definition | SNR = 1/sigma^2 per resource element, E\|H\|^2 = 1, unit-modulus pilots |

## 5. Notes and known deviations

- The reported ChannelNet models use reduced training (4000 training samples, 8-layer DnCNN, 5 + 3 epochs) because of limited compute; the paper uses 32000 samples and a 20-layer DnCNN.
- MMSE uses correlation matrices estimated from all 32000 training channels and the exact SNR (a known-statistics benchmark).
- Estimated MMSE and ALMMSE of the paper are not implemented.
- Test MSE is evaluated on the first 1000 test samples per point by default.
- The simulator in the supplied zip is the TU Wien 5G link-level simulator; only its `FastFading` channel class is used.

# Problem Statement
The doorbell company wants a number instead of "Low": **how many days of battery are left?**
You train a small neural network on your team's log and serve it as a live API from a GitHub
Codespace. The API also reports where its time goes, and whether the model would fit inside
the doorbell itself.

---

## Project Structure
| Path | What it is |
|---|---|
| `data/` | Team training data, such as `team_24_train.csv`. |
| `train.py` | Generates Target 1 plots, removes sensor glitches, evaluates baselines, trains the network, and saves the model. |
| `app/model.py` | Defines the shared `BatteryNet` architecture and its input feature names. |
| `app/main.py` | FastAPI service exposing `/health`, `/model_info`, `/predict`, and `/predict_batch`. |
| `client/bench.py` | Measures API round-trip latency and reported server-stage timings. |
| `model/` | Saved model artifacts: `model.pt` and `config.json`. |
| `plots/` | Generated Target 1 histograms, relationship plots, and linear-prediction plots; ignored by Git. |
| `.devcontainer/` | Codespaces configuration, including Python setup and port forwarding. |
| `requirements.txt` | Python dependencies for training, serving, plotting, and benchmarking. |
| `.gitignore` | Excludes local environments and generated experiment outputs. |

---
## Targets to be achieved

### Target 1: Meet the data and set the bar
- [x] Predict the error of always guessing the average.
- [x] Plot a histogram of every column, and `days_left` against each input. Which relationships are not straight lines?
- [x] **Code (`train.py`):** write the glitch rule for each kind of impossible reading you find, and count the rows removed.
- [x] Report the validation MAE (in days) of **baseline 1**, always predicting the mean.
- [x] **Code (`train.py`):** report the validation MAE of **baseline 2**, linear regression with `np.linalg.lstsq`. Plot predicted against true. Where is it most wrong?
- [x] Run `train.py` as it is and record the network's MAE.

### Target 2: A small network, opened up
- [x] Predict, by hand, the parameters, float32 bytes and MACs of 5 → 32 → 32 → 1.
- [x] **Code (`train.py`):** normalise the inputs using the training rows only. Compare training with and without normalisation.
- [x] Train with hidden widths 4, 8, 16, 32 and 64, with one and two hidden layers (`--hidden 16` or `--hidden 32,32`). Plot validation MAE against parameters.
- [x] Compare training MAE with validation MAE for your chosen size. Is it overfitting?
- [x] Work out the per-layer table (parameters, bytes, MACs) by hand, then **code (`app/main.py`):** make `/model_info` compute it from the network itself.
- [x] Choose your final model and write down the MAE you expect on the company's 400 unseen readings.

### Target 3: Serve it, and time every stage
- [x] Predict the inference time and the laptop-to-Codespace round trip.
- [x] **Code (`app/main.py`):** give every input a range, so glitches get HTTP 422. Base the ranges on what a real sensor can report, not on your data's smallest and largest values: a real reading just outside your data must still get an answer. Test three glitches at `/docs`.
- [x] **Code (`app/main.py`):** time `normalise`, `inference` and `build` the same way `parse` is timed.
- [x] From your **laptop**: `python client/bench.py --url <address> --n 200`. Record the first request, then p50/p95/p99 of the round trip and every stage.
- [x] Draw where the time goes in a typical request (server stages + network and framework). Was the LLM right?
- [x] Run it again with `--no-keepalive`, and once inside the Codespace against `http://localhost:8000`. Explain both differences.
- [x] Time 400 single `/predict` calls against one `/predict_batch` call of 400 rows. Explain the difference.

### Target 4: Would it fit inside the doorbell?
ESP32-S3: about 512 KB SRAM (roughly 300 KB free for your program), 8 MB PSRAM, 16 MB flash.
- [ ] For your model: bytes for weights and biases as float32 and int8, and the largest activation memory needed at once.
- [ ] **Code (`app/main.py`):** add `total_bytes_int8`, `fits_300kb_sram_f32` and `fits_300kb_sram_int8` to `/model_info`.
- [ ] Use the same function to fill in the table for 5 → w → w → 1 at w = 32, 128, 256, 512 and 2048: parameters, float32 size, int8 size, fits in 300 KB SRAM (float32 and int8), fits in 8 MB PSRAM, MACs.
- [ ] Find the exact width where the weights stop fitting in 300 KB, at float32 and at int8.
- [ ] Estimate one prediction's time on the board. State the speed you assume and where it came from.
- [ ] Argue: should the doorbell call your server or run the model itself? Use latency, privacy, cost, power, and what happens when the Wi-Fi is down.

---

## Quick reference

**The API:**
```
GET  /health          -> {"status": "ok", "team": 7, "params": 1281, "val_mae": 3.9}
GET  /model_info      -> {"layers": [{"name": "linear1", "in": 5, "out": 32, "params": 192,
                                      "bytes_f32": 768, "macs": 160}, ...],
                          "total_params": 1281, "total_bytes_f32": 5124, "total_bytes_int8": 1281,
                          "total_macs": 1216, "fits_300kb_sram_f32": true, "fits_300kb_sram_int8": true}
POST /predict         {"battery_v": 3.86, "temp_c": 28.0, "rings_per_day": 5,
                       "wifi_rssi_dbm": -66, "age_months": 22}
                      -> {"days_left": 97.4, "timings_ms": {"parse": .., "normalise": ..,
                          "inference": .., "build": .., "server_total": ..}}
POST /predict_batch   {"rows": [reading, ...]} -> {"days_left": [...], "timings_ms": {...}}
broken reading        -> HTTP 422 with the reason
```

**Columns:** `battery_v` (V), `temp_c` (°C), `rings_per_day`, `wifi_rssi_dbm` (dBm, closer to 0 is stronger), `age_months`, `days_left` (the answer).

**Useful commands**
```
python train.py --team NN --hidden 32,32 --epochs 1500
uvicorn app.main:app --host 0.0.0.0 --port 8000
python client/bench.py --url <address> --n 200 [--no-keepalive]
```

**Counting a `Linear(a, b)` layer:** a×b weights + b biases; a×b MACs per prediction; float32 = 4 bytes, int8 = 1 byte.

---

## Debugging

| Problem | Try |
|---|---|
| The Codespace is still installing | The first start installs PyTorch. Wait; don't rebuild |
| `No module named torch` (or fastapi) | The `.devcontainer` folder was not uploaded. Run `pip install -r requirements.txt` |
| `/health` works in the Codespace but not from your laptop | Port 8000 is still Private. Set it to Public |
| The address stopped working | The Codespace went idle. Start it, rerun `uvicorn`, and set port 8000 to Public again |
| `ModuleNotFoundError: app` | Run commands from the top folder, not from inside `app/` |
| `model/config.json` not found | Run `train.py` first, and commit `model/` |
| `/model_info` is wrong after changing the layer sizes | Compute it from `net.net`, not by hand |
| A result looks impossible | Good: that is the Skeptic's job. Find another explanation and test it |

# 🧠 Neural Resonance Protocol
### Real-Time Inter-Brain Synchronization Game
*Ria & Arushi Agarwal — CONECT Hackathon 2026*

A cooperative two-player game that uses live EEG data from an OpenBCI headset to measure and visualize neural synchrony between players in real time. The game enforces true interdependence — neither player can win alone — and logs inter-brain coherence data for post-session research analysis.

---

## How It Works

Two participants wear a single OpenBCI Ultracortex headset in a split-channel configuration:
- **Channels 0–3** → Person A (F3, C3, P3, O1)
- **Channels 4–7** → Person B (F4, C4, P4, O2)

EEG is streamed via the Lab Streaming Layer (LSL), filtered into alpha (8–12 Hz), beta (13–30 Hz), and theta (4–7 Hz) bands, and combined into a live composite synchrony score. That score directly influences gameplay — higher sync means faster movement and a slower target.

At the end of each 60-second session, the game computes full-session magnitude-squared coherence and exports a timestamped CSV for offline analysis.

---

## Requirements

**Hardware**
- OpenBCI Ultracortex Mark IV headset
- OpenBCI Cyton board

**Python 3.8+**

Install dependencies:
```bash
pip install pygame numpy scipy pylsl
```

> **Note:** `pylsl` and `scipy` are only required for live EEG mode. If no LSL stream is detected, the game automatically falls back to a sine-wave demo signal.

---

## Running the Game

**Step 1 — Start your OpenBCI stream**

Open the OpenBCI GUI, connect your Cyton board, and start an LSL stream before launching the game.

**Step 2 — Launch**
```bash
python game.py
```

**Step 3 — Select difficulty**

| Key | Difficulty | Target Speed | Best For |
|-----|------------|-------------|----------|
| `1` | Slow | 1.8 | Baseline / relaxed |
| `2` | Medium | 3.0 | Standard |
| `3` | Fast | 5.5 | High cognitive load |

**Step 4 — Play**

| Player | Controls |
|--------|----------|
| Player A | `A` / `D` — horizontal movement |
| Player B | `↑` / `↓` — vertical movement |

Overlap your crosshair with the target to capture it. Neither player can reach the target alone.

Press `ESC` at any time to quit.

---

## End Screen

After 60 seconds, the session report screen shows two tabs:

- **Tab `1` — Summary:** Neural coherence breakdown by band, average and peak sync, total captures, and capture-sync correlation analysis
- **Tab `2` — Charts:** Full session sync timeline with capture event markers, and a bar chart comparing sync at captures vs. without

Press `1` or `2` to switch tabs. Press `ESC` to exit.

---

## Data Export

A CSV is automatically saved to the project directory at session end:

```
session_MEDIUM_20260308_143022.csv
```

| Column | Description |
|--------|-------------|
| `time_sec` | Elapsed time in seconds |
| `neural_sync` | Live composite sync score (0–1) |
| `capture` | 1 if a capture occurred at this second, else 0 |
| `difficulty` | Difficulty level selected for the session |

This file is ready for import into Python, R, or any statistical analysis tool.

---

## Architecture

```
OpenBCI Headset
      │
      ▼
Lab Streaming Layer (LSL)
      │
      ▼
BCI Processing Thread (background daemon)
  ├── Bandpass filter (1–40 Hz)
  ├── Per-band Pearson correlation (α, β, θ)
  ├── Composite sync score
  └── UDP broadcast → 127.0.0.1:5005
      │
      ▼
PyGame Main Thread
  ├── Receives sync score via UDP
  ├── Updates gameplay (speed, visuals)
  ├── Logs sync timeline + capture events
  └── End screen: coherence analysis + CSV export
```

---

## Demo Mode

If no LSL stream is found (e.g. no headset connected), the game runs automatically in **demo mode** using a sine-wave signal. All gameplay, visualization, and export features work normally. The BCI status indicator in the top HUD will show `DEMO`.

---

## Project Structure

```
├── game.py              # Main game + BCI processing thread
└── README.md
```

---

## Limitations

- Small sample size — results are proof-of-concept
- Motion artifacts from gameplay are not filtered (no ICA)
- Single channel per brain region per participant
- Pearson correlation used for live score; magnitude-squared coherence computed only at session end

---

## License

Built for academic/research purposes at CONECT Hackathon 2026. All data collected with participant consent. Not intended for clinical use.

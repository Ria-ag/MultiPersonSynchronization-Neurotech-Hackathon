"""
bci_processor.py  —  Neural Resonance Protocol
================================================
Reads TWO EEG streams from OpenBCI GUI via Lab Streaming Layer (LSL),
computes inter-brain alpha coherence in real time, and sends the result
(a float 0.0–1.0) to the game over UDP.

SETUP
-----
1.  Install dependencies:
        pip install pyopenbci pylsl numpy scipy

2.  In OpenBCI GUI:
        - Connect your Cyton board(s)
        - Go to Networking → LSL
        - Add a stream for each headset:
              Stream 1 name : "P1_EEG"   (or leave as default "obci_eeg1")
              Stream 2 name : "P2_EEG"   (or leave as default "obci_eeg2")
        - Press "Start LSL Stream" for both

3.  Run this script BEFORE launching the game:
        python bci_processor.py

    Then launch the game in a separate terminal:
        python neural_resonance.py

CONFIGURATION
-------------
Edit the constants below to match your setup.
"""

import time
import socket
import threading
import numpy as np
from scipy import signal
from pylsl import StreamInlet, resolve_streams

# ─── Configuration ─────────────────────────────────────────────────────────────

# LSL stream names as set in OpenBCI GUI → Networking → LSL
# If you only have ONE Cyton with 2x 4-channel daisy, set SINGLE_BOARD = True
P1_STREAM_NAME  = "obci_eeg1"       # Stream name for Player 1's headset
P2_STREAM_NAME  = "obci_eeg2"       # Stream name for Player 2's headset
SINGLE_BOARD    = False             # True = both players on one 8-channel Cyton
                                    #   P1 = channels 1-4, P2 = channels 5-8

# Which EEG channels to use (0-indexed). Frontal channels are best for
# alpha coherence — Fp1/Fp2 are typically channels 0 and 1 on the Cyton.
P1_CHANNELS     = [0, 1]            # Channel indices to average for P1
P2_CHANNELS     = [0, 1]            # Channel indices to average for P2
                                    # (if SINGLE_BOARD: P2 offset by 4 automatically)

# Sampling & processing
SAMPLE_RATE     = 250               # Cyton default: 250 Hz
WINDOW_SECONDS  = 2.0               # Analysis window length in seconds
WINDOW_SAMPLES  = int(WINDOW_SECONDS * SAMPLE_RATE)   # = 500 samples
UPDATE_HZ       = 10                # How often to recompute sync (times/sec)

# Alpha band (Hz)
ALPHA_LOW       = 8.0
ALPHA_HIGH      = 12.0

# Smoothing: exponential moving average (0=no smooth, 0.9=heavy smooth)
SMOOTHING       = 0.75

# UDP target — must match the game's UDP_PORT
UDP_IP          = "127.0.0.1"
UDP_PORT        = 5005

# Metric: "coherence" or "plv"
# coherence = alpha power correlation  (smoother, easier to explain)
# plv       = phase-locking value      (more sensitive, noisier)
METRIC          = "coherence"

# ─── Globals ───────────────────────────────────────────────────────────────────

p1_buffer = []   # list of scalar values (mean of P1 channels per sample)
p2_buffer = []
buffer_lock = threading.Lock()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ─── Signal processing helpers ─────────────────────────────────────────────────

def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float,
                    fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return signal.filtfilt(b, a, data)


def alpha_coherence(sig1: np.ndarray, sig2: np.ndarray, fs: float) -> float:
    """
    Compute magnitude-squared coherence in the alpha band (8–12 Hz),
    then average across that band.  Returns a value in [0, 1].
    """
    nperseg = min(len(sig1), 128)
    freqs, Cxy = signal.coherence(sig1, sig2, fs=fs, nperseg=nperseg)
    alpha_mask = (freqs >= ALPHA_LOW) & (freqs <= ALPHA_HIGH)
    if not np.any(alpha_mask):
        return 0.0
    return float(np.mean(Cxy[alpha_mask]))


def phase_locking_value(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """
    Phase-Locking Value in the alpha band.
    Filters both signals, extracts instantaneous phase via Hilbert transform,
    then computes PLV = |mean(e^{i * phase_diff})|.  Returns a value in [0, 1].
    """
    f1 = bandpass_filter(sig1, ALPHA_LOW, ALPHA_HIGH, SAMPLE_RATE)
    f2 = bandpass_filter(sig2, ALPHA_LOW, ALPHA_HIGH, SAMPLE_RATE)
    phase1 = np.angle(signal.hilbert(f1))
    phase2 = np.angle(signal.hilbert(f2))
    plv = np.abs(np.mean(np.exp(1j * (phase1 - phase2))))
    return float(plv)


def compute_sync(p1_win: np.ndarray, p2_win: np.ndarray) -> float:
    """Choose metric and return a sync value in [0, 1]."""
    if METRIC == "plv":
        return phase_locking_value(p1_win, p2_win)
    else:
        return alpha_coherence(p1_win, p2_win, SAMPLE_RATE)


# ─── LSL acquisition threads ───────────────────────────────────────────────────

def find_stream(name: str, timeout: float = 10.0) -> StreamInlet:
    """Resolve a named LSL stream and return an inlet."""
    print(f"  Looking for LSL stream '{name}' ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        streams = resolve_streams(wait_time=1.0)
        for s in streams:
            if s.name() == name:
                inlet = StreamInlet(s, max_buflen=30)
                print(f"  ✓ Connected to '{name}'  "
                      f"({s.channel_count()} ch @ {s.nominal_srate()} Hz)")
                return inlet
    raise RuntimeError(
        f"Stream '{name}' not found after {timeout}s.\n"
        f"Make sure OpenBCI GUI is running and LSL streaming is enabled."
    )


def acquisition_thread_dual(p1_name: str, p2_name: str):
    """Acquire from two separate LSL streams (two Cytons)."""
    inlet1 = find_stream(p1_name)
    inlet2 = find_stream(p2_name)
    print("  Both streams connected. Starting acquisition...\n")

    while True:
        # Pull latest available samples (non-blocking)
        samples1, _ = inlet1.pull_chunk(timeout=0.0)
        samples2, _ = inlet2.pull_chunk(timeout=0.0)

        with buffer_lock:
            for s in samples1:
                p1_val = float(np.mean([s[ch] for ch in P1_CHANNELS]))
                p1_buffer.append(p1_val)
            for s in samples2:
                p2_val = float(np.mean([s[ch] for ch in P2_CHANNELS]))
                p2_buffer.append(p2_val)

        time.sleep(1.0 / (SAMPLE_RATE * 2))


def acquisition_thread_single(stream_name: str):
    """Acquire from ONE LSL stream (single Cyton, P1=ch1-4, P2=ch5-8)."""
    inlet = find_stream(stream_name)
    print("  Single-board mode: P1=ch1-4, P2=ch5-8\n")

    while True:
        samples, _ = inlet.pull_chunk(timeout=0.0)
        with buffer_lock:
            for s in samples:
                p1_chs = P1_CHANNELS
                p2_chs = [c + 4 for c in P2_CHANNELS]
                p1_buffer.append(float(np.mean([s[ch] for ch in p1_chs])))
                p2_buffer.append(float(np.mean([s[ch] for ch in p2_chs])))

        time.sleep(1.0 / (SAMPLE_RATE * 2))


# ─── Processing & send loop ────────────────────────────────────────────────────

def processing_loop():
    """
    Every (1 / UPDATE_HZ) seconds:
      1. Grab the last WINDOW_SAMPLES from each buffer
      2. Compute sync metric
      3. Smooth with EMA
      4. Send float over UDP to the game
    """
    smoothed_sync = 0.0
    interval = 1.0 / UPDATE_HZ

    print(f"Processing loop started  "
          f"(metric={METRIC}, window={WINDOW_SECONDS}s, update={UPDATE_HZ}Hz)\n")

    while True:
        time.sleep(interval)

        with buffer_lock:
            n1, n2 = len(p1_buffer), len(p2_buffer)

        if n1 < WINDOW_SAMPLES or n2 < WINDOW_SAMPLES:
            # Not enough data yet — send 0 and wait
            pct1 = int(100 * n1 / WINDOW_SAMPLES)
            pct2 = int(100 * n2 / WINDOW_SAMPLES)
            print(f"\r  Buffering...  P1: {pct1}%  P2: {pct2}%     ", end="", flush=True)
            sock.sendto(b"0.0", (UDP_IP, UDP_PORT))
            continue

        with buffer_lock:
            win1 = np.array(p1_buffer[-WINDOW_SAMPLES:], dtype=np.float64)
            win2 = np.array(p2_buffer[-WINDOW_SAMPLES:], dtype=np.float64)
            # Trim buffers to avoid unbounded growth
            if len(p1_buffer) > WINDOW_SAMPLES * 4:
                del p1_buffer[:-WINDOW_SAMPLES]
            if len(p2_buffer) > WINDOW_SAMPLES * 4:
                del p2_buffer[:-WINDOW_SAMPLES]

        # Compute raw sync
        try:
            raw_sync = compute_sync(win1, win2)
        except Exception as e:
            print(f"\n  [WARN] compute_sync error: {e}")
            raw_sync = 0.0

        raw_sync = float(np.clip(raw_sync, 0.0, 1.0))

        # Exponential moving average
        smoothed_sync = SMOOTHING * smoothed_sync + (1 - SMOOTHING) * raw_sync

        # Send to game
        msg = f"{smoothed_sync:.4f}".encode()
        sock.sendto(msg, (UDP_IP, UDP_PORT))

        print(f"\r  SYNC  raw={raw_sync:.3f}  smoothed={smoothed_sync:.3f}  "
              f"[{'█' * int(smoothed_sync * 20):<20}]  ", end="", flush=True)


# ─── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  NEURAL RESONANCE PROTOCOL  —  BCI Processor")
    print(f"  Metric  : {METRIC.upper()}")
    print(f"  Band    : {ALPHA_LOW}–{ALPHA_HIGH} Hz (alpha)")
    print(f"  Window  : {WINDOW_SECONDS}s  ({WINDOW_SAMPLES} samples)")
    print(f"  Output  : UDP {UDP_IP}:{UDP_PORT}")
    print("=" * 60)

    # Start acquisition in background thread
    if SINGLE_BOARD:
        acq = threading.Thread(
            target=acquisition_thread_single,
            args=(P1_STREAM_NAME,),
            daemon=True
        )
    else:
        acq = threading.Thread(
            target=acquisition_thread_dual,
            args=(P1_STREAM_NAME, P2_STREAM_NAME),
            daemon=True
        )

    acq.start()

    # Processing runs on main thread (blocks here)
    try:
        processing_loop()
    except KeyboardInterrupt:
        print("\n\n  Stopped by user.")


if __name__ == "__main__":
    main()
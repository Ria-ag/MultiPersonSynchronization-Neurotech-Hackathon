from pylsl import StreamInlet, resolve_streams
from scipy.signal import butter, filtfilt, welch
import numpy as np
import socket
import time

# ─── UDP setup (sends sync score to the game) ─────────────────────────────────
UDP_IP   = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ─── Config ───────────────────────────────────────────────────────────────────
fs             = 250       # Cyton sample rate (Hz)
WINDOW_SAMPLES = 1250      # 5s window — same as your original target_samples
UPDATE_EVERY   = 25        # recompute after every N new samples (~10 updates/sec)

# ─── Connect to LSL ───────────────────────────────────────────────────────────
print("Looking for EEG streams...")
streams = resolve_streams()

if len(streams) == 0:
    print("No LSL streams found.")
    exit()

print("Streams found:", len(streams))
inlet = StreamInlet(streams[0])
print("Connected to stream. Collecting data...\n")

# ─── Signal processing (your original functions, unchanged) ───────────────────

def bandpass(data, low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data)

def band_power(signal, low, high):
    freqs, psd = welch(signal, fs, nperseg=min(256, len(signal)))
    return np.mean(psd[(freqs >= low) & (freqs <= high)])

def band_sync(sigA, sigB, low, high):
    A = bandpass(sigA, low, high, fs)
    B = bandpass(sigB, low, high, fs)
    return np.corrcoef(A, B)[0, 1]

# ─── Rolling buffer & main loop ───────────────────────────────────────────────
buffer = []          # grows until WINDOW_SAMPLES, then slides
samples_since_update = 0

while True:
    # Pull latest sample
    sample, timestamp = inlet.pull_sample()
    buffer.append(sample)
    samples_since_update += 1

    # Keep buffer at exactly WINDOW_SAMPLES once full (sliding window)
    if len(buffer) > WINDOW_SAMPLES:
        buffer.pop(0)

    # Only compute when we have a full window AND enough new samples have arrived
    if len(buffer) < WINDOW_SAMPLES or samples_since_update < UPDATE_EVERY:
        continue

    samples_since_update = 0

    # ── Your original processing, verbatim ────────────────────────────────────
    data = np.array(buffer)
    eeg  = data.T                          # shape: (channels, time)

    filtered = np.zeros_like(eeg)
    for ch in range(8):
        filtered[ch] = bandpass(eeg[ch], 1, 40, fs)

    personA = filtered[0:4]
    personB = filtered[4:8]

    alphaA = band_power(personA[0], 8, 12)
    alphaB = band_power(personB[0], 8, 12)

    betaA  = band_power(personA[0], 13, 30)
    betaB  = band_power(personB[0], 13, 30)

    thetaA = band_power(personA[0], 4, 7)
    thetaB = band_power(personB[0], 4, 7)

    alpha_sync = band_sync(personA[0], personB[0], 8, 12)
    beta_sync  = band_sync(personA[0], personB[0], 13, 30)
    theta_sync = band_sync(personA[0], personB[0], 4,  7)

    team_score = (
        0.4 * alpha_sync +
        0.4 * beta_sync  +
        0.2 * theta_sync
    )
    # ── End your original processing ──────────────────────────────────────────

    # corrcoef returns [-1, 1] — clamp to [0, 1] for the game
    team_score_clamped = float(np.clip(team_score, 0.0, 1.0))

    # Send to game
    sock.sendto(str(team_score_clamped).encode(), (UDP_IP, UDP_PORT))

    # Console readout
    bar = '█' * int(team_score_clamped * 30)
    print(f"  α={alpha_sync:+.3f}  β={beta_sync:+.3f}  θ={theta_sync:+.3f}  "
          f"SCORE={team_score_clamped:.3f}  [{bar:<30}]", flush=True)
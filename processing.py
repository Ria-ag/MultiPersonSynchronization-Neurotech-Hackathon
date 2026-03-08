from pylsl import StreamInlet, resolve_streams
from scipy.signal import butter, filtfilt, welch
import numpy as np
import time

print("Looking for EEG streams...")
streams = resolve_streams()

if len(streams) == 0:
    print("No LSL streams found.")
    exit()

print("Streams found:", len(streams))

inlet = StreamInlet(streams[0])
print("Connected to stream. Collecting data...")

samples = []

target_samples = 1250

while len(samples) < target_samples:
    sample, timestamp = inlet.pull_sample()
    samples.append(sample)

data = np.array(samples)
print("Data shape:", data.shape)

# transpose to channels x time
eeg = data.T

fs = 250

def bandpass(data, low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data)

# basic EEG filtering
filtered = np.zeros_like(eeg)
for ch in range(8):
    filtered[ch] = bandpass(eeg[ch], 1, 40, fs)

# split players
personA = filtered[0:4]
personB = filtered[4:8]

def band_power(signal, low, high):
    freqs, psd = welch(signal, fs, nperseg=min(256, len(signal)))
    return np.mean(psd[(freqs >= low) & (freqs <= high)])

def band_sync(sigA, sigB, low, high):
    A = bandpass(sigA, low, high, fs)
    B = bandpass(sigB, low, high, fs)
    return np.corrcoef(A, B)[0,1]

# compute power
alphaA = band_power(personA[0], 8, 12)
alphaB = band_power(personB[0], 8, 12)

betaA = band_power(personA[0], 13, 30)
betaB = band_power(personB[0], 13, 30)

thetaA = band_power(personA[0], 4, 7)
thetaB = band_power(personB[0], 4, 7)

# compute synchronization
alpha_sync = band_sync(personA[0], personB[0], 8, 12)
beta_sync = band_sync(personA[0], personB[0], 13, 30)
theta_sync = band_sync(personA[0], personB[0], 4, 7)

# compatibility score
team_score = (
    0.4 * alpha_sync +
    0.4 * beta_sync +
    0.2 * theta_sync
)

print("\n---- Brain Metrics ----")
print("Alpha Power A:", alphaA)
print("Alpha Power B:", alphaB)

print("Beta Power A:", betaA)
print("Beta Power B:", betaB)

print("Theta Power A:", thetaA)
print("Theta Power B:", thetaB)

print("\n---- Synchronization ----")
print("Alpha Sync:", alpha_sync)
print("Beta Sync:", beta_sync)
print("Theta Sync:", theta_sync)

print("\nTEAM COMPATIBILITY SCORE:", team_score)
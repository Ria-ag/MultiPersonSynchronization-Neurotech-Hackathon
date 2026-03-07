#pip install pylsl numpy scipy matplotlib

from pylsl import StreamInlet, resolve_stream
from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

print("Searching for EEG stream...")

streams = resolve_stream('type','EEG')
inlet = StreamInlet(streams[0])

print("Connected to stream")

plt.ion()
sync_history = []

while True:
    sample, timestamp = inlet.pull_sample()
    
    eeg = sample[0:8]

    personA = eeg[0:4]
    personB = eeg[4:8]

    print("A:", personA, "B:", personB)

    signalA = np.mean(personA)
    signalB = np.mean(personB)

    def bandpass(data, low=8, high=12, fs=250):
        nyq = fs/2
        b,a = butter(4,[low/nyq, high/nyq],btype='band')
        return filtfilt(b,a,data)
    
    sync = np.corrcoef(signalA_window, signalB_window)[0,1]

    sync = compute_sync()

    sync_history.append(sync)

    plt.clf()
    plt.plot(sync_history[-50:])
    plt.ylim(-1,1)
    plt.title("Brain Synchrony")
    plt.pause(0.01)
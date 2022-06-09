# BrainflowCytonEEGWrapper

This is particularly aimed at the OpenBCI Cyton/Cyton + Daisy biosensing board.

It allows for low-level board commands such as configuring channels and setting sample rate, making it easier to configure the way you retrieve data from the board.

## Status

Full functional, working with Cyton and Cyton + Daisy.

Todo:
- [] Include script to process SDCard data (exists but in a different repo)
- [] Make the interface more consistent
- [] Bake in multithreading support for easier parallel data collection + processing
- [] ???

## Details

Default channels coming in with cyton and daisy with the ultracortex mk4 are:

- 0: "pkg"
- 1: "Fp1"
- 2: "Fp2"
- 3: "C3"
- 4: "C4"
- 5: "P7"
- 6: "P8"
- 7: "O1"
- 8: "O2"
- 9: "F7"
- 10: "F8"
- 11: "F3"
- 12: "F4"
- 13: "T7"
- 14: "T8"
- 15: "P3"
- 16: "P4"
- 17: "AX" (accelerometer x)
- 18: "AY" (accelerometer y)
- 19: "AZ" (accelerometer z)
- 31: "marker" (this can be used to put event markers in the EEG data, which is extremely useful, BUT the accelerometer will be disabled)

## Usage examples

### Read data from a dummy board in real time

```python
from BrainflowCyton.eeg import EEG
from time import sleep

eeg_source = EEG(dummyBoard = True)
eeg_source.start_stream(sdcard = False)

while True:
  try:
    sleep(0.5)
    data = eeg_source.poll()
  except KeyboardInterrupt:
    eeg_source.stop()
  
```

### Read data from a real board in real time

```python
from BrainflowCyton.eeg import EEG
from time import sleep

eeg_source = EEG()
eeg_source.start_stream(sdcard = False)

while True:
  try:
    sleep(0.5)
    data = eeg_source.poll()
  except KeyboardInterrupt:
    eeg_source.stop()
  
```

### Set a custom sample rate

*Note: to use sample rates above 250, an SDCard is required, streaming is limited to 250 Hz.*

```python
from BrainflowCyton.eeg import EEG, CytonSampleRate
from time import sleep

eeg_source = EEG()
eeg_source.start_stream(sdcard = True, sr = CytonSampleRate.SR_1000)

while True:
  try:
    sleep(0.5)
    data = eeg_source.poll()
  except KeyboardInterrupt:
    eeg_source.stop()
  
```

### Bandpass the data

```python
from BrainflowCyton.eeg import EEG, Filtering
from time import sleep

eeg_source = EEG(dummyBoard = True)
# set the indexes of channels you want to filter
ch_idx = [1, 2, 3, 4, 5, 6, 7]
eeg_filter = Filtering(exg_channels = ch_idx, sampling_rate=250)

while True:
  try:
    sleep(0.5)
    eeg_source.start_stream(sdcard = False)
    data = eeg_source.poll()
    filtered_data = eeg_filter.bandpass(data, 8, 32)
  except KeyboardInterrupt:
    eeg_source.stop()
  
```

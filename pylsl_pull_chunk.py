"""Example program to demonstrate how to read a multi-channel time-series
from LSL in a chunk-by-chunk manner (which is more efficient)."""

from pylsl import StreamInlet
from pylsl import resolve_streams
import time

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_streams(wait_time=1)

if not streams:
    print("no EEG streams found")
    exit()

print("Available streams:")
for i, stream in enumerate(streams):
    print(f"{i}: {stream.name()}")

# prompt user to select a stream
choice = int(input("Select a stream by index: "))
if choice < 0 or choice >= len(streams):
    print("Invalid choice")
    exit()

# create a new inlet to read from the selected stream
inlet = StreamInlet(streams[choice])

while True:
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    chunk, timestamps = inlet.pull_chunk()
    if timestamps:
        print(len(timestamps), timestamps, chunk)
    time.sleep(1)
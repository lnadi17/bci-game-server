import os
import time
from tempfile import TemporaryFile

import numpy as np
from mne import set_log_level
import mne

from mne_lsl.stream import StreamLSL

set_log_level("WARNING")

def get_stream():
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']

    stream = StreamLSL(bufsize=5, name='UnicornRecorderRawDataLSLStream')
    stream.connect(processing_flags='all', acquisition_delay=0.001)
    stream.pick(picks=['0', '1', '2', '3', '4', '5', '6', '7'])
    stream.rename_channels({
        '0': 'Fz',
        '1': 'C3',
        '2': 'Cz',
        '3': 'C4',
        '4': 'Pz',
        '5': 'PO7',
        '6': 'Oz',
        '7': 'PO8',
    })
    stream.set_montage(montage)
    print(stream.info)
    return stream

def acquisition_loop(stream, save_path=None):
    outfile = TemporaryFile()

    if save_path:
        # Create a folder to save the data (exist_ok=True)
        os.makedirs(save_path, exist_ok=True)

    try:
        while True:
            # Figure how many new samples are available, in seconds
            winsize = stream.n_new_samples / stream.info["sfreq"]
            print(f'New samples (s): {winsize}')
            # Get a new chunk of data
            data, timestamps = stream.get_data(winsize=winsize)
            print(data.shape)

            if save_path:
                if data.size > 0:
                    print(f"Received chunk: {data.shape}")
                    np.save(outfile, data)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Acquisition stopped by user.")
    finally:
        save_as_fif(outfile, save_path)
        outfile.close()

def save_as_fif(outfile, save_path):
    # Go to the start
    _ = outfile.seek(0)

    # Load the full data
    full_data = []

    while True:
        try:
            data = np.load(outfile)
            print(f"Loaded chunk: {data.shape}")
            full_data.append(data)
        except EOFError:
            print("EOFError! No more data to read.")
            break

    print(full_data)
    if save_path:
        print(f"Data saved to {save_path}")


def main():
    stream = get_stream()
    acquisition_loop(stream, save_path='./recordings')

if __name__ == '__main__':
    main()
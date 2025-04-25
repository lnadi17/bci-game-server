import asyncio
import os
import time
import uuid
from tempfile import TemporaryFile
import multiprocessing
from websocket_server import WebSocketServer

import numpy as np
from mne import set_log_level
import mne
from mne import Annotations

from mne_lsl.stream import StreamLSL

set_log_level("WARNING")


def get_stream():
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']

    # stream = StreamLSL(bufsize=10, name='UnicornRecorderRawDataLSLStream')
    stream = StreamLSL(bufsize=10, source_id='testStream')
    stream.connect(processing_flags='all', acquisition_delay=0.001)
    # stream.pick(picks=['0', '1', '2', '3', '4', '5', '6', '7'])
    # stream.rename_channels({
    #     '0': 'Fz',
    #     '1': 'C3',
    #     '2': 'Cz',
    #     '3': 'C4',
    #     '4': 'Pz',
    #     '5': 'PO7',
    #     '6': 'Oz',
    #     '7': 'PO8',
    # })
    stream.set_montage(montage)
    print(stream.info)
    return stream


async def acquisition_loop_async(stream, annotation_list, timestamp_data, window_size=2, save_path=None):
    outfile = TemporaryFile()
    last_read_time = None

    try:
        while True:
            winsize_samples = stream.n_new_samples
            if winsize_samples < window_size * stream.info["sfreq"]:
                if last_read_time is not None:
                    # If we are not getting new data, update the latest timestamp
                    timestamp_data['latest'] = last_read_time + winsize_samples / stream.info["sfreq"]
                await asyncio.sleep(0.01)
                continue

            data, timestamps = stream.get_data(winsize=window_size)
            last_read_time = timestamps[-1]
            timestamp_data['latest'] = last_read_time

            if timestamp_data['first'] is None:
                timestamp_data['first'] = timestamps[0]  # first sample time
                annotation_list.append((timestamps[0], "Stream Started"))  # add start annotation

            if save_path and data.size > 0:
                np.save(outfile, data)
    except asyncio.CancelledError:
        print("Acquisition stopped.")
    finally:
        save_as_fif(outfile, save_path, stream.info, annotation_list, timestamp_data)
        outfile.close()


async def websocket_server_async(annotation_list, timestamp_data):
    server = WebSocketServer()

    async def on_message(websocket, message):
        timestamp = timestamp_data['latest']
        if timestamp is None:
            print("Warning: No timestamp available yet, annotation skipped.")
            return

        print(f"Message from {id(websocket)} at {timestamp}: {message}")
        annotation_list.append((timestamp, message))

        await server.send_to_client(websocket, {"response": "Message received"})

    # Create async callbacks
    async def on_connect(websocket, path):
        print(f"Client connected: {id(websocket)}")

    async def on_disconnect(websocket):
        print(f"Client disconnected: {id(websocket)}")

    server.on_connect(on_connect).on_message(on_message).on_disconnect(on_disconnect)

    await server.start()
    print("WebSocket server started!")

    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        await server.stop()


async def main():
    stream = get_stream()

    # Track both first and latest timestamps
    timestamp_data = {'first': None, 'latest': None}
    annotation_list = []

    await asyncio.gather(
        websocket_server_async(annotation_list, timestamp_data),
        acquisition_loop_async(stream, annotation_list, timestamp_data, window_size=2, save_path='./recordings')
    )


def save_as_fif(outfile, save_path, info, annotation_list, timestamp_data):
    _ = outfile.seek(0)
    full_data = []

    while True:
        try:
            data = np.load(outfile)
            full_data.append(data)
        except EOFError:
            break

    full_data = np.concatenate(full_data, axis=1)
    raw = mne.io.RawArray(full_data, info=info)

    # Create MNE annotations
    if annotation_list and timestamp_data['first'] is not None:
        onsets = []
        durations = []
        descriptions = []

        first_timestamp = timestamp_data['first']
        for timestamp, description in annotation_list:
            # Convert to relative time (in seconds) from recording start
            onset = timestamp - first_timestamp
            onsets.append(onset)
            durations.append(0.0)  # or set if needed
            descriptions.append(description)

        raw.set_annotations(Annotations(onsets, durations, descriptions))

    print(raw.annotations)
    os.makedirs(save_path, exist_ok=True)
    uid = str(uuid.uuid4())[-4:]
    raw.save(os.path.join(save_path, f'recording_{uid}.raw.fif'), overwrite=True)
    print(f"Data saved to {save_path}/recording_{uid}.raw.fif")


if __name__ == '__main__':
    asyncio.run(main())

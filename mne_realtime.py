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


async def acquisition_loop_async(stream, annotation_list, latest_timestamp, window_size=2, save_path=None):
    outfile = TemporaryFile()
    try:
        while True:
            winsize_samples = stream.n_new_samples
            if winsize_samples < window_size * stream.info["sfreq"]:
                await asyncio.sleep(0.05)
                continue

            data, timestamps = stream.get_data(winsize=window_size)

            if timestamps.size > 0:
                latest_timestamp['value'] = timestamps[-1]  # last sample time

            if save_path and data.size > 0:
                np.save(outfile, data)
    except asyncio.CancelledError:
        print("Acquisition stopped.")
    finally:
        save_as_fif(outfile, save_path, stream.info, annotation_list)
        outfile.close()


async def websocket_server_async(annotation_list, latest_timestamp):
    server = WebSocketServer()

    async def on_message(websocket, message):
        timestamp = latest_timestamp['value']
        if timestamp is None:
            print("Warning: No timestamp available yet, annotation skipped.")
            return

        print(f"Message from {id(websocket)} at {timestamp}: {message}")
        annotation_list.append((timestamp, message))

        await server.send_to_client(websocket, {"response": "Message received"})

    server.on_connect(lambda ws, p: print(f"Client connected: {id(ws)}")) \
        .on_message(on_message) \
        .on_disconnect(lambda ws: print(f"Client disconnected: {id(ws)}"))

    await server.start()
    print("WebSocket server started!")

    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        await server.stop()


async def main():
    stream = get_stream()

    # Shared structure for annotations
    latest_timestamp = {'value': None}
    annotation_list = []

    await asyncio.gather(
        websocket_server_async(annotation_list, latest_timestamp),
        acquisition_loop_async(stream, annotation_list, latest_timestamp, window_size=2, save_path='./recordings')
    )


def save_as_fif(outfile, save_path, info, annotation_list):
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
    if annotation_list:
        onsets = []
        durations = []
        descriptions = []

        for timestamp, description in annotation_list:
            # Convert timestamp to relative time (in seconds)
            onset = timestamp - raw.first_time
            onsets.append(onset)
            durations.append(0.0)  # or set if needed
            descriptions.append(description)

        raw.set_annotations(Annotations(onsets, durations, descriptions))

    os.makedirs(save_path, exist_ok=True)
    uid = str(uuid.uuid4())[-4:]
    raw.save(os.path.join(save_path, f'recording_{uid}.raw.fif'), overwrite=True)
    print(f"Data saved to {save_path}")


if __name__ == '__main__':
    asyncio.run(main())

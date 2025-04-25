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


async def acquisition_loop_async(stream, window_size=2, save_path=None):
    outfile = TemporaryFile()
    try:
        while True:
            winsize_samples = stream.n_new_samples
            winsize_time = winsize_samples / stream.info["sfreq"]

            if winsize_time < window_size:
                await asyncio.sleep(0.05)
                continue

            print(f'New samples (s): {winsize_time}')

            # Get a new chunk of data
            data, timestamps = stream.get_data(winsize=window_size)
            print(f'Acquired chunk: {data.shape}')

            if save_path:
                if data.size > 0:
                    print(f"Saving chunk: {data.shape}")
                    np.save(outfile, data)
    except asyncio.CancelledError:
        print("Acquisition stopped.")
    finally:
        save_as_fif(outfile, save_path, stream.info)
        outfile.close()


async def websocket_server_async():
    server = WebSocketServer()

    async def on_connect(websocket, path):
        print(f"Client connected: {id(websocket)}")

    async def on_message(websocket, message):
        print(f"Message from {id(websocket)}: {message}")
        await server.send_to_client(websocket, {"response": "Message received"})

    async def on_disconnect(websocket):
        print(f"Client disconnected: {id(websocket)}")

    server.on_connect(on_connect).on_message(on_message).on_disconnect(on_disconnect)

    await server.start()
    print("WebSocket server started!")
    try:
        await asyncio.Future()  # Run forever
    except asyncio.CancelledError:
        await server.stop()


async def main():
    # Start the LSL stream
    stream = get_stream()

    # Run websocket server and acquisition loop concurrently
    await asyncio.gather(
        websocket_server_async(),
        acquisition_loop_async(stream, window_size=2, save_path='./recordings')
    )


def save_as_fif(outfile, save_path, info):
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

    # Create a fif file based on the full data concatenated along the first axis
    full_data = np.concatenate(full_data, axis=1)
    print(f"Full data shape: {full_data.shape}")

    # Create a folder to save the data (exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # Save the data to a fif file
    uid = str(uuid.uuid4())[-4:]
    mne.io.RawArray(full_data, info=info).save(os.path.join(save_path, f'recording_{uid}.raw.fif'), overwrite=True)

    if save_path:
        print(f"Data saved to {save_path}")


if __name__ == '__main__':
    asyncio.run(main())

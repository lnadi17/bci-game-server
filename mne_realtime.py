import asyncio
import os
import time
import json
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

# Boolean that indicates if the game is playing or not
IS_PLAYING = None
# Context variable to identify which model to train or use (while playing)
# Can be either "ssvep"/"imagery" for training or "ssvep"/"imagery"/"relax" for playing
CONTEXT = None
# List to store annotations
ANNOTATION_LIST = []
# Timestamp variable to track the first and latest timestamps
TIMESTAMP_DATA = {'first': None, 'latest': None}
# Temporary file to store data
OUTFILE = None


def get_stream():
    montage = mne.channels.make_standard_montage("standard_1020")
    stream = StreamLSL(bufsize=10, name='UnicornRecorderRawDataLSLStream')
    stream.connect(processing_flags='all', acquisition_delay=0.001, timeout=10)
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


def get_mock_stream():
    montage = mne.channels.make_standard_montage("standard_1020")
    stream = StreamLSL(bufsize=10, source_id='testStream')
    stream.connect(processing_flags='all', acquisition_delay=0.001, timeout=10)
    stream.set_montage(montage)
    print(stream.info)
    return stream


async def acquisition_loop_async(window_size=2, save_path=None):
    global OUTFILE, TIMESTAMP_DATA, STREAM
    last_read_time = None

    try:
        while True:
            winsize_samples = STREAM.n_new_samples
            if winsize_samples < window_size * STREAM.info["sfreq"]:
                if last_read_time is not None:
                    # If we are not getting new data, update the latest timestamp
                    TIMESTAMP_DATA['latest'] = last_read_time + winsize_samples / STREAM.info["sfreq"]
                await asyncio.sleep(0.01)
                continue

            data, timestamps = STREAM.get_data(winsize=window_size)
            # Update first timestamp (if needed)
            if TIMESTAMP_DATA['first'] is None:
                TIMESTAMP_DATA['first'] = timestamps[0]
                ANNOTATION_LIST.append((TIMESTAMP_DATA['first'], "startRecording"))

            # Update latest timestamp
            last_read_time = timestamps[-1]
            TIMESTAMP_DATA['latest'] = last_read_time

            # Write to OUTFILE if possible
            if save_path and data.size > 0 and OUTFILE is not None:
                np.save(OUTFILE, data)
    except asyncio.CancelledError:
        print("Acquisition stopped.")


def process_realtime_data(data):
    # TODO: Based on context, this function should use a specific model to process the data,
    # and then send this information via websocket
    pass


def parse_annotation(event_name, data):
    timestamp = TIMESTAMP_DATA['latest']
    if timestamp is None:
        print("Warning: No timestamp available yet, annotation skipped.")
        return

    try:
        suffix = ""
        if data.get("frequency"):
            suffix = " " + str(data["frequency"])
        if data.get("classLabel"):
            suffix = " " + str(data["classLabel"])
        annotation = event_name + suffix
        ANNOTATION_LIST.append((timestamp, annotation))
    except KeyError as e:
        print(f"KeyError: {e} in data: {data}")
        return event_name


async def handle_client_message(event_name, data):
    global IS_PLAYING, CONTEXT, ANNOTATION_LIST, OUTFILE, STREAM

    # startTraining
    if event_name == "trainingStarted":
        training_type = data.get("trainingType")
        CONTEXT = training_type
        ANNOTATION_LIST = []
        IS_PLAYING = False
        OUTFILE = TemporaryFile()
        if TIMESTAMP_DATA['latest'] is None:
            print('Warning: No latest timestamp available yet, but training has started.')
            return
        # Update first timestamp
        TIMESTAMP_DATA['first'] = None
    # stopTraining
    elif event_name == "trainingFinished":
        if OUTFILE is None:
            print("Warning: No outfile available, training stopped without saving.")
            return
        # Add annotation for training finished
        ANNOTATION_LIST.append((TIMESTAMP_DATA['latest'], "saveRecording"))
        await asyncio.sleep(4)  # Wait for the last data to buffer be written (just in case)
        # Make sure nothing is written into OUTFILE anymore
        outfile = OUTFILE
        OUTFILE = None
        # Save the data, close the file afterwards
        save_path = save_as_fif(outfile, save_path='./recordings', info=STREAM.info)
        outfile.close()
        # TODO: Train the model with the data here
        #
    elif event_name == "startPlaying":
        IS_PLAYING = True
        # TODO: Start sending messages from realtime model predictions
        pass
    elif event_name == "setContext":
        context = data.get("context")
        if context in ["ssvep", "imagery", "relax"]:
            CONTEXT = context
        else:
            print(f"Unknown context: {context}")
            return
    elif event_name.startswith("cue") or event_name.startswith("stimulus"):
        parse_annotation(event_name, data)
    else:
        print(f"Unknown event name: {event_name}")
        return


async def websocket_server_async():
    global SERVER  # Use the global server variable

    async def on_message(websocket, message):
        timestamp = TIMESTAMP_DATA['latest']
        print(f"Message from {id(websocket)} at {timestamp}: {message}")

        # Extract the required fields
        try:
            parsed = json.loads(message)
            event_name = parsed.get("eventName")
            data = parsed.get("data", {})

            await handle_client_message(event_name, data)

            await SERVER.send_to_client(websocket, {"response": "Message received"})
        except:
            print(f"Error parsing message: {message}")
            await SERVER.send_to_client(websocket, {"error": "Invalid message format"})
            return

    # Create async callbacks
    async def on_connect(websocket, path):
        print(f"Client connected: {id(websocket)}")

    async def on_disconnect(websocket):
        print(f"Client disconnected: {id(websocket)}")

    SERVER.on_connect(on_connect).on_message(on_message).on_disconnect(on_disconnect)

    await SERVER.start()
    print("WebSocket server started!")

    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        await SERVER.stop()


async def main():
    await asyncio.gather(
        websocket_server_async(),
        acquisition_loop_async(window_size=2, save_path='./recordings')
    )


def save_as_fif(outfile, save_path, info):
    _ = outfile.seek(0)
    full_data = []

    while True:
        try:
            data = np.load(outfile)
            full_data.append(data)
        except EOFError:
            break

    info = mne.create_info(info['ch_names'], info['sfreq'], ch_types='eeg')

    full_data = np.concatenate(full_data, axis=1)
    raw = mne.io.RawArray(full_data, info=info)

    # Create MNE annotations
    if ANNOTATION_LIST and TIMESTAMP_DATA['first'] is not None:
        onsets = []
        durations = []
        descriptions = []

        first_timestamp = TIMESTAMP_DATA['first']
        for timestamp, description in ANNOTATION_LIST:
            # Convert to relative time (in seconds) from recording start
            onset = timestamp - first_timestamp
            onsets.append(onset)
            durations.append(0.0)  # or set if needed
            descriptions.append(description)

        raw.set_annotations(Annotations(onsets, durations, descriptions))

    print(raw.annotations)
    os.makedirs(save_path, exist_ok=True)
    uid = str(uuid.uuid4())[-4:]
    full_save_path = os.path.join(save_path, f'recording_{uid}.raw.fif')
    raw.save(full_save_path, overwrite=True)
    print(f"Data saved to {full_save_path}")
    return full_save_path


if __name__ == '__main__':
    # Declare server as a global variable
    SERVER = WebSocketServer()
    # Declare stream as a global variable
    STREAM = get_mock_stream()
    asyncio.run(main())

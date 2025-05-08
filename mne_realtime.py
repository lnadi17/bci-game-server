import asyncio
import os
import time
import json
import uuid
from tempfile import TemporaryFile
import multiprocessing

from processing.RealtimeRelaxModel import RelaxModel
from processing.RealtimeSSVEPModel import SSVEPModel
from processing.RealtimeImageryModel import ImageryModel
from websocket_server import WebSocketServer

import numpy as np
from mne import set_log_level
import mne
from mne import Annotations

from mne_lsl.stream import StreamLSL

set_log_level("WARNING")

# TODO: Frequencies are hardcoded
SSVEP_FREQS = [8, 11, 15]

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
# Global dictionary to store connected clients
CONNECTED_CLIENTS = {}
# Variable for trained models
MODELS = {
    "relax": None,
    "imagery": None,
    "ssvep": None
}
# Variable for latest recordings for different contexts
LATEST_RECORDINGS = {
    "relax": None,
    "imagery": None,
    "ssvep": None
}


def get_raw_stream():
    montage = mne.channels.make_standard_montage("standard_1020")
    stream = StreamLSL(bufsize=10, name='UnicornRecorderRawDataLSLStream')
    stream.connect(processing_flags='all', acquisition_delay=0.001, timeout=2)
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


def get_stream():
    montage = mne.channels.make_standard_montage("standard_1020")
    stream = StreamLSL(bufsize=10, name='UnicornRecorderLSLStream')
    stream.connect(processing_flags='all', acquisition_delay=0.001, timeout=2)
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
    stream.connect(processing_flags='all', acquisition_delay=0.001, timeout=2)
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

            # Use realtime processing if the game is playing
            if IS_PLAYING:
                process_realtime_data(data)

            # Write to OUTFILE if possible
            if save_path and data.size > 0 and OUTFILE is not None:
                np.save(OUTFILE, data)
    except asyncio.CancelledError:
        print("Acquisition stopped.")


def process_realtime_data(data):
    # Based on context, load the model
    if CONTEXT is None:
        print("Warning: No context available, cannot process data.")
        return

    model = MODELS[CONTEXT]

    # Optional: export data to recordings dir
    # uid = str(uuid.uuid4())[-4:]
    # np.save(f'./recordings/data_focus_{uid}', data)

    # Make prediction for the data chunk
    prediction = model.predict(data)
    print(f"Prediction: {prediction}")

    # Send the prediction to all clients
    asyncio.create_task(send_to_all_clients({"currentLabel": prediction}))


def parse_annotation(event_name, data):
    global ANNOTATION_LIST

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


async def handle_client_message(event_name, data):
    global IS_PLAYING, CONTEXT, ANNOTATION_LIST, OUTFILE, STREAM

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
        save_path, uid = save_as_fif(outfile, save_path='./recordings', info=STREAM.info)
        outfile.close()

        # Start training the model
        if CONTEXT is None:
            print("Warning: No context available, training finished but model will not be trained.")
            return

        # Save the path to the latest recordings
        LATEST_RECORDINGS[CONTEXT] = save_path

        # Train the model using the latest recording for the context
        # By default, model is 'auto', and path is latest
        if CONTEXT == "ssvep":
            model = SSVEPModel(SSVEP_FREQS, model_variant='auto', data_path='')
        elif CONTEXT == "imagery":
            model = ImageryModel(model_variant='auto', data_path=LATEST_RECORDINGS[CONTEXT])
        elif CONTEXT == "relax":
            model = RelaxModel(model_variant='auto', data_path=LATEST_RECORDINGS[CONTEXT])
        else:
            print(f"Unknown context: {CONTEXT}")
            return

        # Don't forget to load model (this does the actual training)
        model.load_model()
        MODELS[CONTEXT] = model
        print(f"Important: Model {CONTEXT} trained with data from {LATEST_RECORDINGS[CONTEXT]}!")
        print(f"Model accuracy: {model.accuracy}")
        print(f"Confusion matrix: {model.confusion_matrix}")
    elif event_name == "startPlaying":
        IS_PLAYING = True

        # Load the model based on the data received, it's either 'auto' or the name of the model
        # Same for the data_path, it can be either 'auto' or the path to the data
        model_name = data.get("modelName")
        data_path = data.get("dataPath")
        print(data_path)

        if CONTEXT is None:
            print("Warning: No context available, cannot start playing.")
            return

        if data_path == 'auto':
            # Use the latest recording for the context
            data_path = LATEST_RECORDINGS.get(CONTEXT)

        if CONTEXT == "ssvep":
            model = SSVEPModel(SSVEP_FREQS, model_variant=model_name, data_path='')
        elif CONTEXT == "imagery":
            model = ImageryModel(model_variant=model_name, data_path=data_path)
        elif CONTEXT == "relax":
            model = RelaxModel(model_variant=model_name, data_path=data_path)
        else:
            print(f"Unknown context: {CONTEXT}")
            return

        # Retrain the model on the latest recording
        model.load_model()
        MODELS[CONTEXT] = model
        print(f"Important: Model {CONTEXT} loaded with data from {data_path}!")
        print(f"Model accuracy: {model.accuracy}")
        print(f"Confusion matrix: {model.confusion_matrix}")
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
        # print(f"Message from {id(websocket)} at {timestamp}: {message}")
        print(f"Message from {id(websocket)} at {timestamp}")

        # Extract the required fields
        try:
            parsed = json.loads(message)
            event_name = parsed.get("eventName")
            data = parsed.get("data", {})

            await handle_client_message(event_name, data)

            # await SERVER.send_to_client(websocket, {"response": "Message received"})
        except Exception as e:
            print(f"Error parsing message: {message}")
            print(e)
            await SERVER.send_to_client(websocket, {"error": "Invalid message format"})
            return

    async def on_connect(websocket, path):
        client_id = id(websocket)
        CONNECTED_CLIENTS[client_id] = websocket
        print(f"Client connected: {client_id}")

    async def on_disconnect(websocket):
        client_id = id(websocket)
        CONNECTED_CLIENTS.pop(client_id, None)
        print(f"Client disconnected: {client_id}")

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


async def send_to_all_clients(message):
    for ws in CONNECTED_CLIENTS.values():
        await SERVER.send_to_client(ws, message)


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

    return full_save_path, uid


if __name__ == '__main__':
    # TODO: SSEVP model is hardcoded
    MODELS["ssvep"] = SSVEPModel(SSVEP_FREQS, model_variant='auto', data_path='')
    MODELS["ssvep"].load_model()

    # Declare server as a global variable
    SERVER = WebSocketServer()
    # Declare stream as a global variable
    STREAM = get_stream()
    asyncio.run(main())

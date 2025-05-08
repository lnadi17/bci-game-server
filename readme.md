
# BCI Game Server

A real-time Brain-Computer Interface (BCI) game server for EEG-based experiments and games. This server supports real-time EEG data acquisition, annotation, model training, and prediction using SSVEP, motor imagery, and relaxation paradigms. Communication with clients is handled via WebSockets.

## Features

- **Real-time EEG acquisition** via LSL streams (Unicorn hardware expected)
- **WebSocket server** for communication with game clients
- **Annotation and event handling** for training and playing sessions
- **Model training and prediction** for SSVEP, motor imagery, and relaxation tasks
- **Data saving** in MNE `.fif` format with annotations
- **Modular processing pipeline** for easy extension

## Directory Structure

```
bci-game-server/
├── mne_realtime.py                # Main server logic: acquisition, annotation, model management
├── websocket_server.py            # WebSocket server implementation
├── websocket_client.py            # Example/mock WebSocket client
├── mne_player_lsl.py              # LSL player for testing (sample data)
├── processing/
│   ├── preprocessing.py           # EEG data preprocessing pipeline
│   ├── RealtimeSSVEPModel.py      # SSVEP model (CCA-based)
│   ├── RealtimeImageryModel.py    # Motor imagery model (Riemannian/MDM)
│   └── RealtimeRelaxModel.py      # Relax/focus model (Riemannian/TS)
├── recordings/                    # Saved EEG recordings (created at runtime)
└── readme.md                      # Project documentation
```

## Requirements

- Python 3.8+
- [MNE](https://mne.tools/)
- [numpy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pyriemann](https://pyriemann.readthedocs.io/)
- [websockets](https://websockets.readthedocs.io/)
- [mne-lsl](https://github.com/labstreaminglayer/mne-lsl)

Install dependencies:
```bash
pip install mne numpy scikit-learn pyriemann websockets mne-lsl
```

## Usage

### 0. Start the LSL Stream

Before starting the server, ensure an LSL stream is available:

- **Using Unicorn Hybrid Black**: Start the UnicornRecorder software and configure the desired stream settings.
- **Using Mock Stream**: Run the `mne_player_lsl.py` script to simulate an LSL stream:
```bash
python mne_player_lsl.py
```
You can choose between different LSL streams:

- **Raw Data:** Use get_raw_stream() to access unprocessed data from UnicornRecorder.
- **Prefiltered Data:** Use get_stream() to access prefiltered data based on the selected options in UnicornRecorder.

### 1. Start the Server

Run the main server script:
```bash
python mne_realtime.py
```
This will:
- Start the WebSocket server (default: `ws://localhost:8765`)
- Begin real-time EEG acquisition from the LSL stream

### 2. Connect a Client

Clients (e.g., a game or experiment UI) should connect via WebSocket and send JSON messages to control training, annotation, and prediction.

See `websocket_client.py` for a simple mock client example.

### 3. Workflow

- **Training**: Send `trainingStarted` and `trainingFinished` events with context (`ssvep`, `imagery`, or `relax`). The server records data, saves it, and trains the appropriate model.
- **Playing**: Send `startPlaying` with model/data selection. The server loads the model and streams predictions to all connected clients.
- **Annotations**: Send events like `cue*` or `stimulus*` to annotate the data stream.

### 4. Data & Models

- Recordings are saved in the `recordings/` directory as `.raw.fif` files with MNE annotations.
- Models are trained on the latest recordings and used for real-time predictions.

## Model Details

- **SSVEP**: Canonical Correlation Analysis (CCA) classifier for frequency recognition.
- **Imagery**: Riemannian geometry-based Minimum Distance to Mean (MDM) classifier for left/right hand imagery.
- **Relax**: Tangent Space (TS) classifier for relax/focus detection.

All models use the `processing/preprocessing.py` pipeline for filtering, epoching, and feature extraction.

## Extending

- Add new models by creating a new file in `processing/` and updating `mne_realtime.py`.
- Adjust preprocessing parameters in `processing/preprocessing.py` as needed.

## Development & Testing

- Use `websocket_client.py` to simulate client connections and test server responses.
- The server prints model accuracy and confusion matrix after training.

## Troubleshooting

- Ensure the LSL stream is available and named correctly (see `get_raw_stream()` in `mne_realtime.py`).
- Check Python package versions if you encounter import or compatibility errors.

## License

MIT License

## Acknowledgements

- [MNE-Python](https://mne.tools/)
- [Lab Streaming Layer (LSL)](https://labstreaminglayer.readthedocs.io/)
- [PyRiemann](https://pyriemann.readthedocs.io/)


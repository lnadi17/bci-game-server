import os

import matplotlib.pyplot as plt
import numpy as np
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score, StratifiedKFold
import mne
from processing.preprocessing import BCIDataProcessor


class RelaxModel:
    def __init__(self, model_name: str, recording_path: str = None):
        self.model_name = model_name
        self.recording_path = recording_path
        self.model = None

    def load_model(self):
        # Load the model based on the model name
        if self.model_name == "relax_csp_model":
            self.model = self._load_model1()
        if self.model_name == "relax_old_model":
            self.model = self._load_model2()
        else:
            raise ValueError(f"Model {self.model_name} not recognized.")

    def _load_model2(self):
        # Save current recording path
        current_path = self.recording_path
        self.recording_path = './recordings/recording_relax2.raw.fif'
        trained_model = self._load_model1()
        # Restore original recording path
        self.recording_path = current_path
        return trained_model

    def _load_model1(self):
        window_size = 2
        window_overlap = 0
        processor = BCIDataProcessor(self.recording_path, l_freq=7, h_freq=30, window_size=window_size,
                                     window_overlap=window_overlap)
        data = processor.process()
        # Only select left_hand and right_hand
        data = {label: data[label] for label in data.keys() if label in ['relax', 'focus']}
        X = np.concatenate(list(data.values()), axis=0)
        y = np.concatenate([[label] * data[label].shape[0] for label in data.keys()])  # (samples,)
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Define a monte-carlo cross-validation generator (reduce variance):
        scores = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Assemble a classifier
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

        # Use scikit-learn Pipeline with cross_val_score function
        clf = Pipeline([("CSP", csp), ("LDA", lda)])
        scores = cross_val_score(clf, X_trainval, y_trainval, cv=cv, n_jobs=1)

        # Printing the results
        class_balance = np.mean(y_trainval == "relax")
        class_balance = max(class_balance, 1.0 - class_balance)
        print(f"Classification accuracy: {np.mean(scores)} / Chance level: {class_balance}")

        # Fit on the entire dataset (no train-test split)
        clf.fit(X, y)
        return clf

    def predict(self, data):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Assume data shape is (n_channels, n_times) = (8, 500)
        sfreq = 250  # Set your sampling frequency accordingly
        ch_names = ['Fz','C3','Cz','C4','Pz','PO7','Oz','PO8']
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        # Bandpass filter (same as BCIDataProcessor)
        raw_filtered = raw.copy().filter(l_freq=7, h_freq=30)

        # Get the filtered data as numpy array, shape (n_channels, n_times)
        filtered_data = raw_filtered.get_data()

        # Reshape to (1, n_channels, n_times) for sklearn
        X = filtered_data[np.newaxis, :, :]

        return self.model.predict(X)[0]
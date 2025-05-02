import numpy as np
from pyriemann.classification import TSClassifier
from pyriemann.estimation import Covariances
from sklearn.cross_decomposition import CCA
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from processing.preprocessing import BCIDataProcessor


class CCAClassifier:
    def __init__(self, sampling_rate=250, frequencies=(8, 11, 14), num_targets=3, n_components=1, harmonics=1):
        self.frequencies = frequencies
        self.sampling_rate = sampling_rate
        self.num_targets = num_targets
        self.n_components = n_components
        self.harmonics = harmonics

    def get_reference_signals(self, length, target_freq):
        """Generate sine and cosine templates (1st & 2nd harmonics) for a given frequency."""
        t = np.arange(0, length / self.sampling_rate, 1.0 / self.sampling_rate)
        # Create first harmonic reference signals
        ref = [
            np.sin(2 * np.pi * target_freq * t),
            np.cos(2 * np.pi * target_freq * t),
        ]
        # Create additional reference signals based on self.harmonics
        for harmonic in range(2, self.harmonics + 1):
            ref.append(np.sin(2 * np.pi * target_freq * harmonic * t))
            ref.append(np.cos(2 * np.pi * target_freq * harmonic * t))
        ref = np.array(ref)
        return ref

    def find_corr(self, eeg_data, references, n_components=1):
        """Perform CCA between EEG and reference signals for each frequency."""
        cca = CCA(n_components=n_components)
        result = np.zeros(references.shape[0])

        for i in range(references.shape[0]):
            cca.fit(eeg_data.T, references[i].T)
            X_c, Y_c = cca.transform(eeg_data.T, references[i].T)
            corr = [np.corrcoef(X_c[:, j], Y_c[:, j])[0, 1] for j in range(n_components)]
            result[i] = np.max(corr)

        return result

    def fit(self, X, y):
        # CCA does not require fitting
        pass

    def predict(self, X):
        """Classify the EEG signals using CCA."""
        predictions = []
        for eeg_data in X:
            length = eeg_data.shape[1]
            references = np.array([
                self.get_reference_signals(length, freq)
                for freq in self.frequencies
            ])
            correlations = self.find_corr(eeg_data, references, self.n_components)
            predicted_class = self.frequencies[np.argmax(correlations)]
            predicted_class = f'{predicted_class:.1f}'
            predictions.append(predicted_class)
        return np.array(predictions)


class SSVEPModel:
    available_models = ('auto', 'cca')
    accuracy = 0
    confusion_matrix = None

    def __init__(self, model_variant: str, data_path: str = None):
        if model_variant not in self.available_models:
            raise ValueError(f"Model variant '{model_variant}' is not supported. Choose from {self.available_models}.")
        self.model_name = model_variant
        self.data_path = data_path
        self.model = None

    def load_model(self):
        if self.model_name == 'auto':
            self.model = self._load_cca_model()
        elif self.model_name == 'ts':
            self.model = self._load_cca_model()
        else:
            raise ValueError(f"Model variant '{self.model_name}' is not supported.")

        # Set input and output shapes
        self.input_shape = (8, 500)

    def _load_cca_model(self):
        rescale = True
        window_size = 2
        window_overlap = 0.33
        filter_method = 'iir'
        l_freq, h_freq = 10, 20

        self.processor = BCIDataProcessor(self.data_path, l_freq=l_freq, h_freq=h_freq, window_size=window_size,
                                          window_overlap=window_overlap, rescale=rescale, filter_method=filter_method)
        data = self.processor.process()

        X = np.concatenate(list(data.values()), axis=0)
        y = np.concatenate([[label] * data[label].shape[0] for label in data.keys()])  # (samples,)

        clf = CCAClassifier(sampling_rate=250, frequencies=(8, 11, 15), num_targets=3, n_components=1, harmonics=1)
        clf.fit(X, y)

        y_pred = clf.predict(X)

        self.accuracy = np.mean(y_pred == y)
        self.confusion_matrix = confusion_matrix(y, y_pred)

        return clf

    def predict(self, data):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        processed_chunk = self.processor.process_chunk(data)
        return self.model.predict(processed_chunk)[0]

import numpy as np
from pyriemann.classification import TSClassifier
from pyriemann.estimation import Covariances
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from processing.preprocessing import BCIDataProcessor


class RelaxModel:
    available_models = ('auto', 'ts')
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
            self.model = self._load_ts_model()
        elif self.model_name == 'csp':
            self.model = self._load_ts_model()
        else:
            raise ValueError(f"Model variant '{self.model_name}' is not supported.")

        # Set input and output shapes
        self.input_shape = (8, 500)

    def _load_ts_model(self):
        # Define variables found to work the best with grid search
        rescale = True
        window_size = 2
        window_overlap = 0.33
        filter_method = 'iir'
        l_freq, h_freq = 10, 20

        self.processor = BCIDataProcessor(self.data_path, l_freq=l_freq, h_freq=h_freq, window_size=window_size,
                                          window_overlap=window_overlap, rescale=rescale, filter_method=filter_method)
        data = self.processor.process()
        data = {label: data[label] for label in data.keys() if label in ['relax', 'focus']}

        X = np.concatenate(list(data.values()), axis=0)
        y = np.concatenate([[label] * data[label].shape[0] for label in data.keys()])  # (samples,)

        clf = Pipeline(steps=[('cov', Covariances(estimator='lwf')), ('ts', TSClassifier(metric='riemann'))])
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

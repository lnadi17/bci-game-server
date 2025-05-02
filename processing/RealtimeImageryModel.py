class ImageryModel:
    available_models = ('auto', 'csp')

    def __init__(self, model_variant: str, data_path: str = None):
        if model_variant not in self.available_models:
            raise ValueError(f"Model variant '{model_variant}' is not supported. Choose from {self.available_models}.")
        self.model_name = model_variant
        self.recording_path = data_path
        self.model = None

    def load_model(self):
        # Load the model based on the model name
        if self.model_name == "model1":
            self.model = self._load_model1()
        elif self.model_name == "model2":
            self.model = self._load_model2()
        else:
            raise ValueError(f"Model {self.model_name} not recognized.")

    def _load_model1(self):
        # Load model1
        pass

    def _load_model2(self):
        # Load model2
        pass

    def predict(self, data):
        # Make predictions using the loaded model
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model.predict(data)
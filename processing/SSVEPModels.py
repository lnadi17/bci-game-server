class SSVEPModels:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.input_shape = None
        self.output_shape = None

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
        return self.model.predict(data )
class ModelResultsGenerator(object):
    def __init__(self, trained_model_path, model_ids):
        self.trained_model_path = trained_model_path
        self.model_ids = model_ids

    def generate(self):
        """TODO: for each trained model,
        create metrics and write to Tyra-compatible database"""
        pass

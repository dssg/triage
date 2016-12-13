class ModelResultsGenerator(object):
    def __init__(self, trained_model_path, model_uuids):
        self.trained_model_path = trained_model_path
        self.model_uuids = model_uuids

    def generate(self):
        """TODO: for each trained model,
        create metrics and write to Tyra-compatible database"""
        pass

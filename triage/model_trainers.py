class ModelTrainer(object):
    def __init__(self, features_table, model_config, trained_model_path):
        self.features_table = features_table
        self.model_config = model_config
        self.trained_model_path = trained_model_path

    def train(self):
        """TODO:
        select x_train, x_test, y_train, y_test from features table
        for each model from model config:
            instantiate model
            run
            save pickle to s3 (trained_model_path + '/' + uuid.uuid4() or something)
            return id for each model
        """
        return ['uuid1', 'uuid2']
